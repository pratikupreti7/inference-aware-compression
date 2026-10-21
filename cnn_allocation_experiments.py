#!/usr/bin/env python
"""
Does bound allocation computed with the FFNN pipeline transfer to a different model family?
Re-uses the per-fold allocations chosen in results_alloc/results_alloc.json (uniform / feature-distortion /
classifier-aware, selected on training subjects only) and evaluates each with the 1-D CNN of cnn_experiments.py
(trained on training subjects reconstructed with the same per-group bounds, tested on the held-out subject).

Usage: python cnn_allocation_experiments.py --wesad_dir data/WESAD --alloc results_alloc/results_alloc.json --out results_cnn_alloc
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wesad_compression_experiments as W
import bound_allocation_experiments as A
import cnn_experiments as C
import torch

def to32_group(g, streams):
    if g == "BVP":
        x = streams["BVP"]; return x[: len(x) // 2 * 2].reshape(-1, 2).mean(axis=1)[None, :]
    if g in ("EDA", "TEMP"):
        return np.repeat(streams[g], 8)[None, :]
    return np.stack([streams["ACC_x"], streams["ACC_y"], streams["ACC_z"]])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad_dir", required=True); ap.add_argument("--alloc", default="results_alloc/results_alloc.json")
    ap.add_argument("--out", default="results_cnn_alloc"); ap.add_argument("--targets", default="18,24"); ap.add_argument("--methods", default="uniform,feat,clf")
    ap.add_argument("--seeds", type=int, default=2); ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rewrite_tex", action="store_true")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); t0 = time.time()
    if args.rewrite_tex:
        write_outputs(json.load(open(os.path.join(args.out, "results.json"))), args.out, print); return
    AL = json.load(open(args.alloc)); levels = AL["meta"]["levels"]; sigma = AL["sigma"]; groups = AL["meta"]["groups"]
    targets = [int(t) for t in args.targets.split(",")]; methods = args.methods.split(",")
    subjects = AL["meta"]["subjects"][:3] if args.quick else AL["meta"]["subjects"]
    seeds = 1 if args.quick else args.seeds; epochs = 2 if args.quick else args.epochs
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    logf = open(os.path.join(args.out, "run.log"), "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    P(f"device={device} subjects={subjects} targets={targets} methods={methods} seeds={seeds} epochs={epochs}")
    # ---- precompute per subject / group / level window tensors
    X = {}; Y = {}
    for sid in subjects:
        ch, label, dur = W.load_subject(args.wesad_dir, sid)
        t0s, y = A.select_windows(label, dur); Y[sid] = y; X[sid] = {}
        for g in groups:
            X[sid][g] = {}
            for li, lv in enumerate(levels):
                streams = {}
                for c in A.GROUP_CH[g]:
                    if lv == 0.0:
                        streams[c] = ch[c]
                    else:
                        eps = lv * sigma[g]; xh, b, err, _ = W.compress_stream(ch[c], "proposed", eps, W.FS[g]); assert err <= eps * (1 + 1e-9); streams[c] = xh
                s32 = to32_group(g, streams)
                X[sid][g][li] = np.stack([s32[:, t * C.FS_OUT:(t + W.WIN_SEC) * C.FS_OUT] for t in t0s]).astype(np.float32)
        P(f"S{sid}: {len(y)} windows ({time.time()-t0:.0f}s)")
    def assemble(sid, combo):
        return np.concatenate([X[sid][g][combo[i]] for i, g in enumerate(groups)], axis=1)
    # ---- evaluate
    res = []
    for sid in subjects:
        for rt in targets:
            for m in methods:
                combo = AL["fold_plans"][f"S{sid}"][str(rt)][m]
                Xtr = np.concatenate([assemble(s, combo) for s in subjects if s != sid]); ytr = np.concatenate([Y[s] for s in subjects if s != sid])
                Xte = assemble(sid, combo); yte = Y[sid]
                for seed in range(seeds):
                    yp = C.train_eval(Xtr, ytr, [Xte], seed, device, epochs)[0]
                    mt = W.metrics(yte, yp)
                    res.append({"sid": sid, "rt": rt, "m": m, "seed": seed, "combo": combo, "acc": mt["acc"], "f1": mt["f1"],
                                "ratio": AL["results"][str(rt)][m]["per_subject_ratio"][f"S{sid}"]})
                P(f"S{sid} rho>={rt} {m} combo={''.join(str(c) for c in combo)}: acc=" + "/".join(f"{100*r['acc']:.1f}" for r in res if r['sid'] == sid and r['rt'] == rt and r['m'] == m) + f" ({time.time()-t0:.0f}s)")
    # ---- aggregate
    R = {"meta": {"subjects": subjects, "targets": targets, "methods": methods, "seeds": seeds, "epochs": epochs, "runtime_s": time.time() - t0, "alloc_source": args.alloc}, "results": {}}
    for rt in targets:
        R["results"][str(rt)] = {}
        for m in methods:
            per = {s: np.mean([r["acc"] for r in res if r["sid"] == s and r["rt"] == rt and r["m"] == m]) for s in subjects}
            perf = {s: np.mean([r["f1"] for r in res if r["sid"] == s and r["rt"] == rt and r["m"] == m]) for s in subjects}
            ratios = {s: AL["results"][str(rt)][m]["per_subject_ratio"][f"S{s}"] for s in subjects}
            v = np.array(list(per.values()))
            R["results"][str(rt)][m] = {"acc_mean": float(v.mean()), "acc_ci95": W.ci95(v), "f1_mean": float(np.mean(list(perf.values()))),
                                        "ratio": AL["results"][str(rt)][m]["ratio"], "per_subject_acc": {f"S{s}": float(per[s]) for s in subjects}}
        for m in methods:
            if m != "uniform" and "uniform" in methods:
                a = np.array([R["results"][str(rt)][m]["per_subject_acc"][f"S{s}"] for s in subjects]); u = np.array([R["results"][str(rt)]["uniform"]["per_subject_acc"][f"S{s}"] for s in subjects])
                R["results"][str(rt)][m]["vs_uniform"] = W.paired_stats(a, u)
    if "uniform" in methods:
        for m in methods:
            if m == "uniform": continue
            d = [R["results"][str(rt)][m]["per_subject_acc"][f"S{s}"] - R["results"][str(rt)]["uniform"]["per_subject_acc"][f"S{s}"] for rt in targets for s in subjects]
            from scipy import stats
            R.setdefault("pooled", {})[m] = {"mean_diff": float(np.mean(d)), "n": len(d), "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if np.any(np.array(d) != 0) else 1.0}
    json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    write_outputs(R, args.out, P)

def write_outputs(R, out, P):
    _D = str.maketrans("0123456789", "ZABCDEFGHI")
    lab = {"uniform": "Uniform $\\epsilon$", "feat": "Feature-distortion alloc.", "clf": "Classifier-aware alloc. (FFNN)"}
    L = []
    def macro(n, v): L.append(f"\\newcommand{{\\cnnalloc{str(n).translate(_D)}}}{{{v}}}\n")
    rows = ["\\begin{tabular}{llcccc}\n\\toprule\n", "Target $\\rho_t$ & Method & $\\rho$ & CNN Acc. (\\%) & $\\Delta$ vs.\\ uniform & $p$ \\\\\n\\midrule\n"]
    for rt in R["meta"]["targets"]:
        for m in R["meta"]["methods"]:
            r = R["results"][str(rt)][m]; v = r.get("vs_uniform")
            rows.append(f"$\\ge{rt}$ & {lab[m]} & {r['ratio']:.1f}$\\times$ & {100*r['acc_mean']:.1f} $\\pm$ {100*r['acc_ci95']:.1f} & " + (f"{100*v['mean_diff']:+.1f} & {A.pfmt(v['p_wilcoxon'])}" if v else "--- & ---") + " \\\\\n")
            macro(f"{m.capitalize()}Acc{rt}", f"{100*r['acc_mean']:.1f}"); macro(f"{m.capitalize()}Ratio{rt}", f"{r['ratio']:.1f}")
            if v:
                macro(f"{m.capitalize()}Diff{rt}", f"{100*v['mean_diff']:+.1f}"); macro(f"{m.capitalize()}P{rt}", A.pfmt(v['p_wilcoxon']))
    rows.append("\\bottomrule\n\\end{tabular}\n")
    for m, v in R.get("pooled", {}).items():
        macro(f"{m.capitalize()}PooledDiff", f"{100*v['mean_diff']:+.1f}"); macro(f"{m.capitalize()}PooledP", f"{v['p_wilcoxon']:.3f}"); macro(f"{m.capitalize()}PooledN", v["n"])
    macro("NSeeds", R["meta"]["seeds"]); macro("Epochs", R["meta"]["epochs"])
    open(os.path.join(out, "table_cnn_alloc.tex"), "w").writelines(rows); open(os.path.join(out, "macros_cnn_alloc.tex"), "w").writelines(L)
    S = [f"CNN with FFNN-derived allocations ({len(R['meta']['subjects'])} subjects, {R['meta']['seeds']} seeds, {R['meta']['epochs']} epochs, {R['meta']['runtime_s']/60:.1f} min)"]
    for rt in R["meta"]["targets"]:
        for m in R["meta"]["methods"]:
            r = R["results"][str(rt)][m]; v = r.get("vs_uniform", {})
            S.append(f"  rho>={rt:<3} {m:<8} ratio={r['ratio']:5.1f} acc={100*r['acc_mean']:5.1f} " + (f"diff={100*v['mean_diff']:+.1f} p={v['p_wilcoxon']:.3f}" if v else ""))
    for m, v in R.get("pooled", {}).items():
        S.append(f"  pooled {m}: {100*v['mean_diff']:+.2f} (p={v['p_wilcoxon']:.4f}, n={v['n']})")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(S) + "\n"); P("\n" + "\n".join(S))

if __name__ == "__main__":
    main()
