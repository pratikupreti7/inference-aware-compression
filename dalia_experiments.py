#!/usr/bin/env python
"""
Second dataset / second task: PPG-DaLiA (Reiss et al., Sensors 2019) - same Empatica E4 wrist channels
(BVP 64 Hz, EDA 4 Hz, TEMP 4 Hz, ACC 32 Hz), 15 subjects, 8 daily activities (IDs 1..8; 0 = transient, dropped).
Same codec, same feature extractor, same window protocol (60 s / 10 s), same FFNN (multi-class), LOSO, 5 seeds.
Configs: raw, proposed at the eps grid (eps_c = eps * sigma_c with sigma from this corpus), uniform quantization,
ZFP, downsampling, lossless. Writes results_dalia/{results.json, macros_dalia.tex, table_dalia.tex, summary.txt}.

Usage: python dalia_experiments.py --dalia_dir data/DALIA --out results_dalia [--quick]
"""
import argparse, json, math, os, sys, time
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wesad_compression_experiments as W

ACT_FS = 4
ACT_NAMES = {1: "sitting", 2: "stairs", 3: "table soccer", 4: "cycling", 5: "driving", 6: "lunch", 7: "walking", 8: "working"}
SUBJECTS = list(range(1, 16))

def load_subject(d, sid):
    z = np.load(os.path.join(d, f"S{sid}.npz"))
    ch = {"BVP": z["BVP"].astype(np.float64), "EDA": z["EDA"].astype(np.float64), "TEMP": z["TEMP"].astype(np.float64),
          "ACC_x": z["ACC"][:, 0].astype(np.float64), "ACC_y": z["ACC"][:, 1].astype(np.float64), "ACC_z": z["ACC"][:, 2].astype(np.float64)}
    act = z["activity"].astype(np.int16)
    dur = int(math.floor(min([len(act) / ACT_FS] + [len(ch[c]) / W.FS[W.CH_GROUP[c]] for c in ch])))
    for c in ch:
        ch[c] = ch[c][: dur * W.FS[W.CH_GROUP[c]]]
    return ch, act[: dur * ACT_FS], dur

def window_features(ch, act, dur, win=W.WIN_SEC, step=W.STEP_SEC):
    X, y, names = [], [], None
    for t0 in range(0, dur - win + 1, step):
        lw = act[t0 * ACT_FS:(t0 + win) * ACT_FS]
        vals, cnts = np.unique(lw, return_counts=True)
        maj = int(vals[np.argmax(cnts)]); purity = cnts.max() / len(lw)
        if maj == 0 or purity < W.MIN_PURITY:
            continue
        f = {}
        f.update(W.feat_bvp(ch["BVP"][t0 * 64:(t0 + win) * 64])); f.update(W.feat_eda(ch["EDA"][t0 * 4:(t0 + win) * 4]))
        f.update(W.feat_temp(ch["TEMP"][t0 * 4:(t0 + win) * 4]))
        f.update(W.feat_acc(ch["ACC_x"][t0 * 32:(t0 + win) * 32], ch["ACC_y"][t0 * 32:(t0 + win) * 32], ch["ACC_z"][t0 * 32:(t0 + win) * 32]))
        if names is None:
            names = list(f.keys())
        X.append([f[k] for k in names]); y.append(maj)
    return np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0), np.array(y, dtype=int), names

def fit_eval_mc(Xtr, ytr, Xte_list, seed, hidden=(64, 32)):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(ytr, return_counts=True); m = counts.max()
    idx = np.concatenate([rng.choice(np.where(ytr == c)[0], m, replace=True) for c in classes]); rng.shuffle(idx)
    sc = StandardScaler().fit(Xtr[idx])
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=400, random_state=seed)
    clf.fit(sc.transform(Xtr[idx]), ytr[idx])
    return [clf.predict(sc.transform(Xte)) for Xte in Xte_list]

def metrics_mc(y, yp):
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": float(accuracy_score(y, yp)), "f1": float(f1_score(y, yp, average="macro"))}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dalia_dir", required=True); ap.add_argument("--out", default="results_dalia")
    ap.add_argument("--seeds", type=int, default=5); ap.add_argument("--quick", action="store_true"); ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--rewrite_tex", action="store_true")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); t_start = time.time()
    if args.rewrite_tex:
        write_tex(json.load(open(os.path.join(args.out, "results.json"))), args.out); print("tex regenerated"); return
    subjects = SUBJECTS[:3] if args.quick else SUBJECTS
    eps_grid = [0.01, 0.1] if args.quick else W.EPS_GRID
    seeds = 1 if args.quick else args.seeds
    logf = open(os.path.join(args.out, "run.log"), "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    data = {}; acc = {g: [0.0, 0.0, 0] for g in W.FS}
    for sid in subjects:
        ch, act, dur = load_subject(args.dalia_dir, sid); data[sid] = (ch, act, dur)
        for c in W.CHANNELS:
            g = W.CH_GROUP[c]; v = ch[c]; acc[g][0] += v.sum(); acc[g][1] += (v ** 2).sum(); acc[g][2] += len(v)
        P(f"loaded S{sid}: {dur} s, activities {dict(sorted(Counter(act.tolist()).items()))}")
    sigma = {g: math.sqrt(max(acc[g][1] / acc[g][2] - (acc[g][0] / acc[g][2]) ** 2, 0.0)) for g in W.FS}
    P("sigma:", sigma)
    configs = [("raw", None)]
    for e in eps_grid:
        configs += [("proposed", e), ("uniform", e)] + ([("zfp", e)] if W.HAVE_ZFP else [])
    configs += [("downsample", D) for D in W.DS_GRID] + [("lossless", None)]
    cname = lambda m, p: m if p is None else f"{m}_{p:g}"
    comp, feats, fnames = {}, {}, None
    for m, p in configs:
        name = cname(m, p); comp[name] = {"method": m, "param": p, "bits": 0.0, "raw_bits": 0.0, "max_err_ratio": 0.0, "bound_ok": True}; feats[name] = {}
        for sid in subjects:
            ch, act, dur = data[sid]; rec = {}
            for c in W.CHANNELS:
                x = ch[c]; fs = W.FS[W.CH_GROUP[c]]
                if m == "raw":
                    xh, bits, err, eps_abs = x.copy(), W.RAW_BITS * len(x), 0.0, None
                else:
                    eps_abs = p * sigma[W.CH_GROUP[c]] if m in ("proposed", "uniform", "zfp") else None
                    xh, bits, err, _ = W.compress_stream(x, m, eps_abs if eps_abs is not None else p, fs)
                    if eps_abs is not None:
                        comp[name]["max_err_ratio"] = max(comp[name]["max_err_ratio"], err / eps_abs)
                        if err > eps_abs * (1 + 1e-9):
                            comp[name]["bound_ok"] = False
                rec[c] = xh; comp[name]["bits"] += bits; comp[name]["raw_bits"] += W.RAW_BITS * len(x)
            X, y, fnames = window_features(rec, act, dur); feats[name][sid] = (X, y)
        comp[name]["ratio"] = comp[name]["raw_bits"] / comp[name]["bits"]; comp[name]["bits_per_sample"] = W.RAW_BITS / comp[name]["ratio"]
        P(f"[{name}] ratio={comp[name]['ratio']:.2f}x max_err/eps={comp[name]['max_err_ratio']:.4f} ok={comp[name]['bound_ok']} ({time.time()-t_start:.0f}s)")
    viol = [n for n, c in comp.items() if c["method"] in ("proposed", "uniform") and not c["bound_ok"]]
    if viol:
        P(f"ERROR: bound violated by {viol}"); raise SystemExit(2)
    n_win = sum(len(feats["raw"][s][1]) for s in subjects); cls_counts = Counter(np.concatenate([feats["raw"][s][1] for s in subjects]).tolist())
    P(f"windows: {n_win}, class counts {dict(sorted(cls_counts.items()))}")
    from joblib import Parallel, delayed
    names = list(comp.keys())
    def job(name, sid, seed):
        Xtr = np.vstack([feats[name][s][0] for s in subjects if s != sid]); ytr = np.concatenate([feats[name][s][1] for s in subjects if s != sid])
        if name == "raw":
            preds = fit_eval_mc(Xtr, ytr, [feats[n][sid][0] for n in names], seed)
            return {"name": name, "sid": sid, "matched": metrics_mc(feats[name][sid][1], preds[names.index("raw")]), "mismatched": {n: metrics_mc(feats[n][sid][1], preds[i]) for i, n in enumerate(names)}}
        preds = fit_eval_mc(Xtr, ytr, [feats[name][sid][0]], seed)
        return {"name": name, "sid": sid, "matched": metrics_mc(feats[name][sid][1], preds[0])}
    jobs = [(n, s, k) for n in names for s in subjects for k in range(seeds)]
    P(f"running {len(jobs)} LOSO fits ..."); res = Parallel(n_jobs=args.n_jobs)(delayed(job)(*j) for j in jobs); P(f"done ({time.time()-t_start:.0f}s)")
    per = {n: {s: {"acc": [], "f1": []} for s in subjects} for n in names}; per_mm = {n: {s: {"acc": [], "f1": []} for s in subjects} for n in names}
    for r in res:
        for k in ("acc", "f1"):
            per[r["name"]][r["sid"]][k].append(r["matched"][k])
        if r["name"] == "raw":
            for n in names:
                for k in ("acc", "f1"):
                    per_mm[n][r["sid"]][k].append(r["mismatched"][n][k])
    def summarize(tab):
        out = {}; ref = {k: np.array([np.mean(tab["raw"][s][k]) for s in subjects]) for k in ("acc", "f1")}
        for n in names:
            out[n] = {}
            for k in ("acc", "f1"):
                v = np.array([np.mean(tab[n][s][k]) for s in subjects])
                out[n][k] = {"mean": float(v.mean()), "ci95": W.ci95(v), "per_subject": {f"S{s}": float(x) for s, x in zip(subjects, v)}, "vs_raw": W.paired_stats(v, ref[k])}
        return out
    R = {"meta": {"date": time.strftime("%Y-%m-%d %H:%M"), "subjects": subjects, "seeds": seeds, "eps_grid": eps_grid, "n_windows": n_win, "class_counts": {str(k): v for k, v in cls_counts.items()},
                  "n_features": len(fnames), "runtime_s": time.time() - t_start, "activities": ACT_NAMES},
         "sigma": sigma, "compression": comp, "classification_matched": summarize(per), "classification_mismatched": summarize(per_mm)}
    json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    write_tex(R, args.out); write_summary(R, args.out, P); P(f"total {time.time()-t_start:.0f}s")

def pfmt(p):
    return "$<$0.01" if p < 0.005 else f"{p:.2f}"

def write_tex(R, out):
    comp, M, U = R["compression"], R["classification_matched"], R["classification_mismatched"]
    key = {0.005: "Ea", 0.01: "Eb", 0.02: "Ec", 0.05: "Ed", 0.1: "Ee", 0.2: "Ef"}
    L = []
    _D = str.maketrans("0123456789", "ZABCDEFGHI")
    def macro(n, v): L.append(f"\\newcommand{{\\{str(n).translate(_D)}}}{{{v}}}\n")
    macro("daliaNWindows", R["meta"]["n_windows"]); macro("daliaNSubjects", len(R["meta"]["subjects"]))
    macro("daliaRawAcc", f"{100*M['raw']['acc']['mean']:.1f}"); macro("daliaRawF", f"{100*M['raw']['f1']['mean']:.1f}"); macro("daliaRawAccCI", f"{100*M['raw']['acc']['ci95']:.1f}")
    for g, s in R["sigma"].items():
        macro(f"daliaSigma{g}", f"{s:.3g}")
    for n, c in comp.items():
        if c["method"] == "proposed":
            k = key.get(c["param"], "Ex")
            macro(f"daliaRatio{k}", f"{c['ratio']:.1f}"); macro(f"daliaAcc{k}", f"{100*M[n]['acc']['mean']:.1f}"); macro(f"daliaDAcc{k}", f"{100*M[n]['acc']['vs_raw']['mean_diff']:+.1f}")
            macro(f"daliaP{k}", pfmt(M[n]['acc']['vs_raw']['p_wilcoxon'])); macro(f"daliaF{k}", f"{100*M[n]['f1']['mean']:.1f}"); macro(f"daliaDF{k}", f"{100*M[n]['f1']['vs_raw']['mean_diff']:+.1f}")
            macro(f"daliaAccMM{k}", f"{100*U[n]['acc']['mean']:.1f}")
        if c["method"] == "downsample":
            macro(f"daliaDsP{c['param']}", pfmt(M[n]['acc']['vs_raw']['p_wilcoxon'])); macro(f"daliaDsPrel{c['param']}", ("<0.01" if M[n]['acc']['vs_raw']['p_wilcoxon'] < 0.005 else f"={M[n]['acc']['vs_raw']['p_wilcoxon']:.2f}")); macro(f"daliaDsAcc{c['param']}", f"{100*M[n]['acc']['mean']:.1f}"); macro(f"daliaDsDAcc{c['param']}", f"{100*M[n]['acc']['vs_raw']['mean_diff']:+.1f}"); macro(f"daliaDsAccMM{c['param']}", f"{100*U[n]['acc']['mean']:.1f}")
    macro("daliaRatioLossless", f"{comp['lossless']['ratio']:.2f}")
    zf = [n for n in comp if comp[n]["method"] == "zfp"]
    if zf:
        macro("daliaZfpRatioMax", f"{max(comp[n]['ratio'] for n in zf):.1f}"); macro("daliaZfpErrMin", f"{min(comp[n]['max_err_ratio'] for n in zf):.2f}"); macro("daliaZfpErrMax", f"{max(comp[n]['max_err_ratio'] for n in zf):.2f}")
    open(os.path.join(out, "macros_dalia.tex"), "w").writelines(L)
    rows = ["\\begin{tabular}{llcccccc}\n\\toprule\n", "Method & Setting & Ratio $\\rho$ & Acc. (\\%) & $\\Delta$Acc. & $p$ & F1 (\\%) & $\\Delta$F1 \\\\\n\\midrule\n"]
    rows.append(f"Uncompressed & --- & 1.0$\\times$ & {100*M['raw']['acc']['mean']:.1f} $\\pm$ {100*M['raw']['acc']['ci95']:.1f} & --- & --- & {100*M['raw']['f1']['mean']:.1f} & --- \\\\\n")
    lab = {"proposed": "Proposed", "uniform": "Scalar quant.", "zfp": "ZFP", "downsample": "Downsampling", "lossless": "Lossless zlib"}
    for n, c in comp.items():
        if c["method"] == "raw" or c["method"] in ("uniform", "zfp"):
            continue
        st = "---" if c["param"] is None else (f"$D={c['param']}$" if c["method"] == "downsample" else f"$\\epsilon={c['param']:g}\\sigma$")
        rows.append(f"{lab[c['method']]} & {st} & {c['ratio']:.1f}$\\times$ & {100*M[n]['acc']['mean']:.1f} $\\pm$ {100*M[n]['acc']['ci95']:.1f} & {100*M[n]['acc']['vs_raw']['mean_diff']:+.1f} & {pfmt(M[n]['acc']['vs_raw']['p_wilcoxon'])} & {100*M[n]['f1']['mean']:.1f} & {100*M[n]['f1']['vs_raw']['mean_diff']:+.1f} \\\\\n")
    rows.append("\\bottomrule\n\\end{tabular}\n"); open(os.path.join(out, "table_dalia.tex"), "w").writelines(rows)

def write_summary(R, out, P):
    comp, M, U = R["compression"], R["classification_matched"], R["classification_mismatched"]
    S = [f"PPG-DaLiA activity recognition ({len(R['meta']['subjects'])} subjects, {R['meta']['seeds']} seeds, {R['meta']['n_windows']} windows, {R['meta']['runtime_s']/60:.1f} min)", f"sigma: {R['sigma']}", f"classes: {R['meta']['class_counts']}"]
    S.append(f"{'config':<16}{'ratio':>7}{'maxerr/eps':>11}{'acc':>7}{'dAcc':>7}{'p':>6}{'F1':>7}{'dF1':>7}{'acc_mm':>8}")
    for n, c in comp.items():
        S.append(f"{n:<16}{c['ratio']:>7.2f}{c['max_err_ratio']:>11.4f}{100*M[n]['acc']['mean']:>7.1f}{100*M[n]['acc']['vs_raw']['mean_diff']:>+7.1f}{M[n]['acc']['vs_raw']['p_wilcoxon']:>6.2f}{100*M[n]['f1']['mean']:>7.1f}{100*M[n]['f1']['vs_raw']['mean_diff']:>+7.1f}{100*U[n]['acc']['mean']:>8.1f}")
    txt = "\n".join(S); open(os.path.join(out, "summary.txt"), "w").write(txt + "\n"); P("\n" + txt)

if __name__ == "__main__":
    main()
