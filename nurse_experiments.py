#!/usr/bin/env python
"""
Third dataset: nurse stress dataset (Hosseini et al., Sci. Data 2022) - Empatica E4 worn by 15 nurses during
hospital shifts (in the wild), self-reported stress events (0 none / 1 medium / 2 high) from survey intervals.
Task: binary stress (level 1 or 2) vs. no stress (0) on 60 s windows that lie inside a labeled interval.
Same codec, features, FFNN, LOSO over nurses, 5 seeds. Headline metric macro-F1 (strong class imbalance).
Usage: python nurse_experiments.py --nurse_dir data/NURSE --out results_nurse [--quick]
"""
import argparse, json, math, os, sys, time
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wesad_compression_experiments as W
import dalia_experiments as DL

LAB_FS = 4

def nurse_ids(d):
    return sorted(f[:-4] for f in os.listdir(d) if f.endswith(".npz"))

def load_subject(d, nid):
    z = np.load(os.path.join(d, f"{nid}.npz"))
    # kept as float32 to fit ~1250 h of data in RAM; converted to float64 per channel when used
    ch = {"BVP": z["BVP"].astype(np.float32), "EDA": z["EDA"].astype(np.float32), "TEMP": z["TEMP"].astype(np.float32),
          "ACC_x": np.ascontiguousarray(z["ACC"][:, 0]).astype(np.float32), "ACC_y": np.ascontiguousarray(z["ACC"][:, 1]).astype(np.float32), "ACC_z": np.ascontiguousarray(z["ACC"][:, 2]).astype(np.float32)}
    lab = z["label"].astype(np.int16)
    dur = int(math.floor(min([len(lab) / LAB_FS] + [len(ch[c]) / W.FS[W.CH_GROUP[c]] for c in ch])))
    for c in ch:
        ch[c] = ch[c][: dur * W.FS[W.CH_GROUP[c]]]
    return ch, lab[: dur * LAB_FS], dur

def select_windows(lab, dur, win=W.WIN_SEC, step=W.STEP_SEC):
    t0s, ys = [], []
    for t0 in range(0, dur - win + 1, step):
        lw = lab[t0 * LAB_FS:(t0 + win) * LAB_FS]
        if (lw < 0).any():
            continue
        vals, cnts = np.unique(lw, return_counts=True); maj = int(vals[np.argmax(cnts)])
        if cnts.max() / len(lw) < W.MIN_PURITY:
            continue
        t0s.append(t0); ys.append(1 if maj > 0 else 0)
    return np.array(t0s, dtype=int), np.array(ys, dtype=int)

def window_features(ch, lab, dur):
    t0s, y = select_windows(lab, dur); X = []; names = None
    for t0 in t0s:
        f = {}
        f.update(W.feat_bvp(ch["BVP"][t0 * 64:(t0 + 60) * 64])); f.update(W.feat_eda(ch["EDA"][t0 * 4:(t0 + 60) * 4]))
        f.update(W.feat_temp(ch["TEMP"][t0 * 4:(t0 + 60) * 4]))
        f.update(W.feat_acc(ch["ACC_x"][t0 * 32:(t0 + 60) * 32], ch["ACC_y"][t0 * 32:(t0 + 60) * 32], ch["ACC_z"][t0 * 32:(t0 + 60) * 32]))
        if names is None: names = list(f.keys())
        X.append([f[k] for k in names])
    X = np.nan_to_num(np.array(X, dtype=np.float64).reshape(len(t0s), -1), nan=0.0, posinf=0.0, neginf=0.0)
    return X, y, names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nurse_dir", required=True); ap.add_argument("--out", default="results_nurse")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--quick", action="store_true"); ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--rewrite_tex", action="store_true")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); t_start = time.time()
    if args.rewrite_tex:
        write_tex(json.load(open(os.path.join(args.out, "results.json"))), args.out); print("tex regenerated"); return
    ids = nurse_ids(args.nurse_dir); ids = ids[:3] if args.quick else ids
    eps_grid = [0.01, 0.1] if args.quick else W.EPS_GRID; seeds = 1 if args.quick else args.seeds
    logf = open(os.path.join(args.out, "run.log"), "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    data = {}; acc = {g: [0.0, 0.0, 0] for g in W.FS}
    for nid in ids:
        ch, lab, dur = load_subject(args.nurse_dir, nid); data[nid] = (ch, lab, dur)
        for c in W.CHANNELS:
            g = W.CH_GROUP[c]; v = ch[c].astype(np.float64); acc[g][0] += v.sum(); acc[g][1] += (v ** 2).sum(); acc[g][2] += len(v)
        P(f"loaded {nid}: {dur/3600:.1f} h, labeled s {dict(sorted(Counter(lab[lab>=0].tolist()).items()))}")
    sigma = {g: math.sqrt(max(acc[g][1] / acc[g][2] - (acc[g][0] / acc[g][2]) ** 2, 0.0)) for g in W.FS}
    P("sigma:", sigma)
    # keep nurses with both classes present (needed for per-subject metrics)
    configs = [("raw", None)]
    for e in eps_grid:
        configs += [("proposed", e), ("uniform", e)] + ([("zfp", e)] if W.HAVE_ZFP else [])
    configs += [("downsample", D) for D in W.DS_GRID] + [("lossless", None)]
    cname = lambda m, p: m if p is None else f"{m}_{p:g}"
    comp, feats, fnames = {}, {}, None
    for m, p in configs:
        name = cname(m, p); comp[name] = {"method": m, "param": p, "bits": 0.0, "raw_bits": 0.0, "max_err_ratio": 0.0, "bound_ok": True}; feats[name] = {}
        for nid in ids:
            ch, lab, dur = data[nid]; rec = {}
            for c in W.CHANNELS:
                x = ch[c].astype(np.float64); fs = W.FS[W.CH_GROUP[c]]
                if m == "raw":
                    xh, bits, err, eps_abs = x, W.RAW_BITS * len(x), 0.0, None
                else:
                    eps_abs = p * sigma[W.CH_GROUP[c]] if m in ("proposed", "uniform", "zfp") else None
                    xh, bits, err, _ = W.compress_stream(x, m, eps_abs if eps_abs is not None else p, fs)
                    if eps_abs is not None:
                        comp[name]["max_err_ratio"] = max(comp[name]["max_err_ratio"], err / eps_abs)
                        if err > eps_abs * (1 + 1e-9): comp[name]["bound_ok"] = False
                rec[c] = xh; comp[name]["bits"] += bits; comp[name]["raw_bits"] += W.RAW_BITS * len(x)
            X, y, fnames = window_features(rec, lab, dur); feats[name][nid] = (X, y)
        comp[name]["ratio"] = comp[name]["raw_bits"] / comp[name]["bits"]; comp[name]["bits_per_sample"] = W.RAW_BITS / comp[name]["ratio"]
        P(f"[{name}] ratio={comp[name]['ratio']:.2f}x max_err/eps={comp[name]['max_err_ratio']:.4f} ok={comp[name]['bound_ok']} ({time.time()-t_start:.0f}s)")
    viol = [n for n, c in comp.items() if c["method"] in ("proposed", "uniform") and not c["bound_ok"]]
    if viol:
        P(f"ERROR: bound violated by {viol}"); raise SystemExit(2)
    for n, c in comp.items():
        if not c["bound_ok"]: P(f"NOTE: baseline {n} exceeded its nominal tolerance (max err/eps = {c['max_err_ratio']:.3f}); reported, not aborted")
    subjects = [n for n in ids if len(set(feats["raw"][n][1].tolist())) == 2 and len(feats["raw"][n][1]) >= 10]
    P(f"nurses with both classes and >=10 windows: {subjects} (of {ids})")
    n_win = sum(len(feats["raw"][s][1]) for s in subjects); n_pos = sum(int(feats["raw"][s][1].sum()) for s in subjects)
    P(f"windows: {n_win}, stress {n_pos} ({100*n_pos/max(n_win,1):.1f}%)")
    from joblib import Parallel, delayed
    names = list(comp.keys())
    def job(name, sid, seed):
        tr = [s for s in ids if s != sid]   # train on all other nurses incl. single-class ones
        Xtr = np.vstack([feats[name][s][0] for s in tr if len(feats[name][s][1])]); ytr = np.concatenate([feats[name][s][1] for s in tr if len(feats[name][s][1])])
        if name == "raw":
            preds = W.fit_eval(Xtr, ytr, [feats[n][sid][0] for n in names], seed)
            return {"name": name, "sid": sid, "matched": W.metrics(feats[name][sid][1], preds[names.index("raw")]), "mismatched": {n: W.metrics(feats[n][sid][1], preds[i]) for i, n in enumerate(names)}}
        preds = W.fit_eval(Xtr, ytr, [feats[name][sid][0]], seed)
        return {"name": name, "sid": sid, "matched": W.metrics(feats[name][sid][1], preds[0])}
    jobs = [(n, s, k) for n in names for s in subjects for k in range(seeds)]
    P(f"running {len(jobs)} LOSO fits ..."); res = Parallel(n_jobs=args.n_jobs)(delayed(job)(*j) for j in jobs); P(f"done ({time.time()-t_start:.0f}s)")
    per = {n: {s: {"acc": [], "f1": []} for s in subjects} for n in names}; per_mm = {n: {s: {"acc": [], "f1": []} for s in subjects} for n in names}
    for r in res:
        for k in ("acc", "f1"): per[r["name"]][r["sid"]][k].append(r["matched"][k])
        if r["name"] == "raw":
            for n in names:
                for k in ("acc", "f1"): per_mm[n][r["sid"]][k].append(r["mismatched"][n][k])
    def summarize(tab):
        out = {}; ref = {k: np.array([np.mean(tab["raw"][s][k]) for s in subjects]) for k in ("acc", "f1")}
        for n in names:
            out[n] = {}
            for k in ("acc", "f1"):
                v = np.array([np.mean(tab[n][s][k]) for s in subjects])
                out[n][k] = {"mean": float(v.mean()), "ci95": W.ci95(v), "per_subject": {s: float(x) for s, x in zip(subjects, v)}, "vs_raw": W.paired_stats(v, ref[k])}
        return out
    R = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M"), "subjects": subjects, "all_ids": ids, "seeds": seeds, "eps_grid": eps_grid, "n_windows": n_win, "n_stress_windows": n_pos,
                  "n_features": len(fnames), "runtime_s": time.time() - t_start}, "sigma": sigma, "compression": comp,
         "classification_matched": summarize(per), "classification_mismatched": summarize(per_mm)}
    json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=1); write_tex(R, args.out)
    M = R["classification_matched"]; U = R["classification_mismatched"]
    S = [f"NURSE stress ({len(subjects)} nurses evaluated of {len(ids)}, {seeds} seeds, {n_win} windows, {100*n_pos/max(n_win,1):.1f}% stress, {R['meta']['runtime_s']/60:.1f} min)", f"sigma: {sigma}",
         f"{'config':<16}{'ratio':>7}{'maxerr/eps':>11}{'acc':>7}{'dAcc':>7}{'p':>6}{'F1':>7}{'dF1':>7}{'pF1':>6}{'F1_mm':>7}"]
    for n, c in comp.items():
        S.append(f"{n:<16}{c['ratio']:>7.2f}{c['max_err_ratio']:>11.4f}{100*M[n]['acc']['mean']:>7.1f}{100*M[n]['acc']['vs_raw']['mean_diff']:>+7.1f}{M[n]['acc']['vs_raw']['p_wilcoxon']:>6.2f}{100*M[n]['f1']['mean']:>7.1f}{100*M[n]['f1']['vs_raw']['mean_diff']:>+7.1f}{M[n]['f1']['vs_raw']['p_wilcoxon']:>6.2f}{100*U[n]['f1']['mean']:>7.1f}")
    open(os.path.join(args.out, "summary.txt"), "w").write("\n".join(S) + "\n"); P("\n" + "\n".join(S))

def write_tex(R, out):
    comp, M, U = R["compression"], R["classification_matched"], R["classification_mismatched"]
    key = {0.005: "Ea", 0.01: "Eb", 0.02: "Ec", 0.05: "Ed", 0.1: "Ee", 0.2: "Ef"}; _D = str.maketrans("0123456789", "ZABCDEFGHI")
    L = []
    def macro(n, v): L.append(f"\\newcommand{{\\nurse{str(n).translate(_D)}}}{{{v}}}\n")
    macro("NSubjects", len(R["meta"]["subjects"])); macro("NAll", len(R["meta"]["all_ids"])); macro("NWindows", R["meta"]["n_windows"]); macro("PctStress", f"{100*R['meta']['n_stress_windows']/max(R['meta']['n_windows'],1):.1f}")
    macro("RawAcc", f"{100*M['raw']['acc']['mean']:.1f}"); macro("RawF", f"{100*M['raw']['f1']['mean']:.1f}"); macro("RawFCI", f"{100*M['raw']['f1']['ci95']:.1f}")
    for g, s in R["sigma"].items(): macro(f"Sigma{g}", f"{s:.3g}")
    for n, c in comp.items():
        if c["method"] == "proposed":
            k = key.get(c["param"], "Ex"); macro(f"Ratio{k}", f"{c['ratio']:.1f}"); macro(f"DAcc{k}", f"{100*M[n]['acc']['vs_raw']['mean_diff']:+.1f}"); macro(f"DF{k}", f"{100*M[n]['f1']['vs_raw']['mean_diff']:+.1f}")
            macro(f"PF{k}", DL.pfmt(M[n]['f1']['vs_raw']['p_wilcoxon'])); macro(f"F{k}", f"{100*M[n]['f1']['mean']:.1f}"); macro(f"FMM{k}", f"{100*U[n]['f1']['mean']:.1f}")
        if c["method"] == "downsample":
            macro(f"DsDF{c['param']}", f"{100*M[n]['f1']['vs_raw']['mean_diff']:+.1f}"); macro(f"DsPF{c['param']}", DL.pfmt(M[n]['f1']['vs_raw']['p_wilcoxon'])); macro(f"DsFMM{c['param']}", f"{100*U[n]['f1']['mean']:.1f}")
    macro("RatioLossless", f"{comp['lossless']['ratio']:.2f}")
    zf = [n for n in comp if comp[n]["method"] == "zfp"]
    if zf:
        worst = max(zf, key=lambda n: comp[n]["max_err_ratio"])
        macro("ZfpErrMax", f"{comp[worst]['max_err_ratio']:.2f}"); macro("ZfpErrMaxEps", f"{comp[worst]['param']:g}"); macro("ZfpRatioMax", f"{max(comp[n]['ratio'] for n in zf):.1f}")
    open(os.path.join(out, "macros_nurse.tex"), "w").writelines(L)
    rows = ["\\begin{tabular}{llcccccc}\n\\toprule\n", "Method & Setting & Ratio $\\rho$ & Acc. (\\%) & F1 (\\%) & $\\Delta$F1 & $p$ & F1 mism. (\\%) \\\\\n\\midrule\n"]
    lab = {"proposed": "Proposed", "downsample": "Downsampling", "lossless": "Lossless zlib"}
    rows.append(f"Uncompressed & --- & 1.0$\\times$ & {100*M['raw']['acc']['mean']:.1f} & {100*M['raw']['f1']['mean']:.1f} $\\pm$ {100*M['raw']['f1']['ci95']:.1f} & --- & --- & {100*U['raw']['f1']['mean']:.1f} \\\\\n")
    for n, c in comp.items():
        if c["method"] not in lab: continue
        st = "---" if c["param"] is None else (f"$D={c['param']}$" if c["method"] == "downsample" else f"$\\epsilon={c['param']:g}\\sigma$")
        rows.append(f"{lab[c['method']]} & {st} & {c['ratio']:.1f}$\\times$ & {100*M[n]['acc']['mean']:.1f} & {100*M[n]['f1']['mean']:.1f} $\\pm$ {100*M[n]['f1']['ci95']:.1f} & {100*M[n]['f1']['vs_raw']['mean_diff']:+.1f} & {DL.pfmt(M[n]['f1']['vs_raw']['p_wilcoxon'])} & {100*U[n]['f1']['mean']:.1f} \\\\\n")
    rows.append("\\bottomrule\n\\end{tabular}\n"); open(os.path.join(out, "table_nurse.tex"), "w").writelines(rows)

if __name__ == "__main__":
    main()
