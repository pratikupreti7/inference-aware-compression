#!/usr/bin/env python
"""
Inference-aware error-bound allocation across channel groups (BVP, EDA, TEMP, ACC) on WESAD.

For each LOSO fold, using TRAINING subjects only:
  R_g(l)  = transmitted bits of group g at bound level l   (rate curve, label-free)
  D_g(l)  = distortion of group g at level l, two variants:
            feat : mean squared change of standardized window features of group g  (label-free)
            clf  : mean squared change of the logit of a classifier trained on uncompressed training
                   features when only group g's features are replaced by their level-l version
Distortions are assumed additive across groups; all level combinations are enumerated and, for each
target ratio rho_t, the combination with minimum total distortion among those whose training-set
ratio >= rho_t is selected. Baseline 'uniform' uses the same bound level for every group (the
scheme of the main experiment) selected by the same rule. The held-out subject is then compressed
with the chosen per-group bounds, a classifier is trained on the training subjects compressed the
same way (matched scenario), and accuracy / achieved ratio are measured on the held-out subject.

Usage: python bound_allocation_experiments.py --wesad_dir data/WESAD --out results_alloc [--quick]
"""
import argparse, itertools, json, math, os, sys, time
import numpy as np
from scipy import stats
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wesad_compression_experiments as W

GROUPS = ["BVP", "EDA", "TEMP", "ACC"]
GROUP_CH = {"BVP": ["BVP"], "EDA": ["EDA"], "TEMP": ["TEMP"], "ACC": ["ACC_x", "ACC_y", "ACC_z"]}
LEVELS = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]     # eps / sigma_g ; 0 = uncompressed
TARGETS = [4, 6, 8, 10, 12, 15, 18, 21, 24, 28]
METHODS = ["uniform", "feat", "clf"]

def select_windows(label, dur, win=W.WIN_SEC, step=W.STEP_SEC):
    t0s, ys = [], []
    for t0 in range(0, dur - win + 1, step):
        lw = label[t0 * W.LABEL_FS:(t0 + win) * W.LABEL_FS]
        vals, cnts = np.unique(lw, return_counts=True)
        maj = int(vals[np.argmax(cnts)]); purity = cnts.max() / len(lw)
        if maj in W.LABELS_KEEP and purity >= W.MIN_PURITY:
            t0s.append(t0); ys.append(1 if maj == 2 else 0)
    return np.array(t0s), np.array(ys, dtype=int)

def select_windows_dalia(act, dur, win=W.WIN_SEC, step=W.STEP_SEC):
    import dalia_experiments as DL
    t0s, ys = [], []
    for t0 in range(0, dur - win + 1, step):
        lw = act[t0 * DL.ACT_FS:(t0 + win) * DL.ACT_FS]
        vals, cnts = np.unique(lw, return_counts=True)
        maj = int(vals[np.argmax(cnts)]); purity = cnts.max() / len(lw)
        if maj != 0 and purity >= W.MIN_PURITY:
            t0s.append(t0); ys.append(maj)
    return np.array(t0s), np.array(ys, dtype=int)

def group_block(g, streams, t0s, win=W.WIN_SEC):
    rows, names = [], None
    for t0 in t0s:
        if g == "BVP":
            f = W.feat_bvp(streams["BVP"][t0 * 64:(t0 + win) * 64])
        elif g == "EDA":
            f = W.feat_eda(streams["EDA"][t0 * 4:(t0 + win) * 4])
        elif g == "TEMP":
            f = W.feat_temp(streams["TEMP"][t0 * 4:(t0 + win) * 4])
        else:
            f = W.feat_acc(streams["ACC_x"][t0 * 32:(t0 + win) * 32], streams["ACC_y"][t0 * 32:(t0 + win) * 32], streams["ACC_z"][t0 * 32:(t0 + win) * 32])
        if names is None:
            names = list(f.keys())
        rows.append([f[k] for k in names])
    X = np.nan_to_num(np.array(rows, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return X, names

def fit_model(Xtr, ytr, seed):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    classes, counts = np.unique(ytr, return_counts=True); mx = counts.max()
    idx = np.concatenate([rng.choice(np.where(ytr == c)[0], mx, replace=True) for c in classes]); rng.shuffle(idx)
    sc = StandardScaler().fit(Xtr[idx])
    clf = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", alpha=1e-3, learning_rate_init=1e-3, max_iter=400, random_state=seed)
    clf.fit(sc.transform(Xtr[idx]), ytr[idx])
    return sc, clf

def logits(sc, clf, X):
    """Log-probability vector (binary: equivalent up to a constant to the logit; multi-class: log-softmax)."""
    p = np.clip(clf.predict_proba(sc.transform(X)), 1e-6, 1 - 1e-6)
    return np.log(p)

def assemble(F, sid, combo):
    return np.hstack([F[sid][g][combo[i]] for i, g in enumerate(GROUPS)])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad_dir", required=True)
    ap.add_argument("--out", default="results_alloc")
    ap.add_argument("--main_results", default="results/results.json")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--dataset", choices=["wesad", "dalia", "nurse"], default="wesad")
    ap.add_argument("--rewrite", action="store_true", help="regenerate outputs from <out>/results_alloc.json")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.rewrite:
        out = json.load(open(os.path.join(args.out, "results_alloc.json")))
        write_outputs(out, args.out, args.main_results, print); return
    t_start = time.time()
    if args.dataset == "dalia":
        import dalia_experiments as DL
        all_subjects = DL.SUBJECTS
    elif args.dataset == "nurse":
        import nurse_experiments as NU
        all_subjects = json.load(open(args.main_results))["meta"]["subjects"]   # nurses with both classes
    else:
        all_subjects = W.SUBJECTS_ALL
    subjects = all_subjects[:4] if args.quick else all_subjects
    levels = [0.0, 0.01, 0.1, 1.0] if args.quick else LEVELS
    targets = [6, 12, 24] if args.quick else TARGETS
    seeds = 1 if args.quick else args.seeds
    logf = open(os.path.join(args.out, "run.log"), "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()

    # ---- load + sigma (identical definition to the main experiment)
    data = {}; acc = {g: [0.0, 0.0, 0] for g in GROUPS}
    for sid in subjects:
        ch, label, dur = (DL.load_subject(args.wesad_dir, sid) if args.dataset == "dalia" else NU.load_subject(args.wesad_dir, sid) if args.dataset == "nurse" else W.load_subject(args.wesad_dir, sid))
        data[sid] = (ch, label, dur)
        for c in W.CHANNELS:
            g = W.CH_GROUP[c]; v = np.asarray(ch[c], dtype=np.float64)
            acc[g][0] += v.sum(); acc[g][1] += (v ** 2).sum(); acc[g][2] += len(v)
    sigma = {g: math.sqrt(max(acc[g][1] / acc[g][2] - (acc[g][0] / acc[g][2]) ** 2, 0.0)) for g in GROUPS}
    if args.dataset == "nurse":
        sigma = {g: float(v) for g, v in json.load(open(args.main_results))["sigma"].items()}
    P("sigma:", sigma)

    # ---- precompute per subject / group / level: bits and feature blocks
    F = {}; B = {}; RAW = {}; Y = {}; names = {}
    for sid in subjects:
        ch, label, dur = data[sid]
        t0s, y = (select_windows_dalia(label, dur) if args.dataset == "dalia" else NU.select_windows(label, dur) if args.dataset == "nurse" else select_windows(label, dur)); Y[sid] = y
        F[sid] = {}; B[sid] = {}; RAW[sid] = {}
        for g in GROUPS:
            F[sid][g] = {}; B[sid][g] = {}
            RAW[sid][g] = sum(W.RAW_BITS * len(ch[c]) for c in GROUP_CH[g])
            for li, lv in enumerate(levels):
                streams = {}; bits = 0
                for c in GROUP_CH[g]:
                    xc = np.asarray(ch[c], dtype=np.float64)
                    if lv == 0.0:
                        streams[c] = xc; bits += W.RAW_BITS * len(xc)
                    else:
                        eps = lv * sigma[g]
                        xh, b, err, _ = W.compress_stream(xc, "proposed", eps, W.FS[g])
                        assert err <= eps * (1 + 1e-9), "bound violated"
                        streams[c] = xh; bits += b
                X, nm = group_block(g, streams, t0s); names[g] = nm
                F[sid][g][li] = X; B[sid][g][li] = bits
        P(f"S{sid}: {len(y)} windows precomputed ({time.time()-t_start:.0f}s)")
    nlev = len(levels)
    combos = list(itertools.product(range(nlev), repeat=len(GROUPS)))
    P(f"{len(combos)} allocations enumerated")

    # ---- per-fold allocation
    from joblib import Parallel, delayed
    fold_plans = {}
    for sid in subjects:
        train = [t for t in subjects if t != sid]
        R = {g: np.array([sum(B[t][g][li] for t in train) for li in range(nlev)], float) for g in GROUPS}
        Rraw = sum(RAW[t][g] for t in train for g in GROUPS)
        # feature distortion (label-free)
        Dfeat = {}
        for g in GROUPS:
            X0 = np.vstack([F[t][g][0] for t in train]); sd = X0.std(axis=0); ok = sd > 1e-12
            Dfeat[g] = np.array([np.mean(((np.vstack([F[t][g][li] for t in train])[:, ok] - X0[:, ok]) / sd[ok]) ** 2) for li in range(nlev)])
        # classifier-aware distortion
        Xtr0 = np.vstack([assemble(F, t, (0,) * len(GROUPS)) for t in train]); ytr = np.concatenate([Y[t] for t in train])
        sc, clf = fit_model(Xtr0, ytr, 0); L0 = logits(sc, clf, Xtr0)
        Dclf = {}
        for gi, g in enumerate(GROUPS):
            vals = []
            for li in range(nlev):
                combo = [0] * len(GROUPS); combo[gi] = li
                Xl = np.vstack([assemble(F, t, tuple(combo)) for t in train])
                vals.append(np.mean((logits(sc, clf, Xl) - L0) ** 2))
            Dclf[g] = np.array(vals)
        ratio_tr = np.array([Rraw / sum(R[g][c[i]] for i, g in enumerate(GROUPS)) for c in combos])
        dfeat = np.array([sum(Dfeat[g][c[i]] for i, g in enumerate(GROUPS)) for c in combos])
        dclf = np.array([sum(Dclf[g][c[i]] for i, g in enumerate(GROUPS)) for c in combos])
        plans = {}
        for rt in targets:
            feas = ratio_tr >= rt
            if not feas.any():
                feas = ratio_tr >= ratio_tr.max() - 1e-9
            uni = [i for i, c in enumerate(combos) if len(set(c)) == 1 and feas[i]]
            plans[rt] = {
                "uniform": combos[min(uni, key=lambda i: ratio_tr[i])] if uni else combos[int(np.argmax(ratio_tr))],
                "feat": combos[int(np.where(feas)[0][np.argmin(dfeat[feas])])],
                "clf": combos[int(np.where(feas)[0][np.argmin(dclf[feas])])],
            }
        fold_plans[sid] = {"plans": plans, "Dfeat": {g: Dfeat[g].tolist() for g in GROUPS}, "Dclf": {g: Dclf[g].tolist() for g in GROUPS},
                           "R_bits_per_sample": {g: (R[g] / (sum(RAW[t][g] for t in train) / W.RAW_BITS)).tolist() for g in GROUPS}}
        P(f"fold S{sid}: plans " + "; ".join(f"rho>={rt}: " + ",".join(f"{m}={''.join(str(l) for l in plans[rt][m])}" for m in METHODS) for rt in targets))

    # ---- evaluation jobs
    def job(sid, rt, m, seed):
        combo = tuple(fold_plans[sid]["plans"][rt][m])
        train = [t for t in subjects if t != sid]
        Xtr = np.vstack([assemble(F, t, combo) for t in train]); ytr = np.concatenate([Y[t] for t in train])
        Xte = assemble(F, sid, combo); yte = Y[sid]
        sc, clf = fit_model(Xtr, ytr, seed)
        yp = clf.predict(sc.transform(Xte))
        bits = sum(B[sid][g][combo[i]] for i, g in enumerate(GROUPS)); raw = sum(RAW[sid][g] for g in GROUPS)
        from sklearn.metrics import accuracy_score, f1_score
        return {"sid": sid, "rt": rt, "m": m, "seed": seed, "combo": combo, "bits": bits, "raw": raw,
                "acc": float(accuracy_score(yte, yp)), "f1": float(f1_score(yte, yp, average="macro"))}
    jobs = [(s, rt, m, k) for s in subjects for rt in targets for m in METHODS for k in range(seeds)]
    P(f"running {len(jobs)} fits ...")
    res = Parallel(n_jobs=args.n_jobs)(delayed(job)(*j) for j in jobs)
    P(f"fits done ({time.time()-t_start:.0f}s)")

    # ---- aggregate
    out = {"meta": {"dataset": args.dataset, "levels": levels, "targets": targets, "methods": METHODS, "subjects": subjects, "seeds": seeds, "groups": GROUPS,
                    "n_combos": len(combos), "runtime_s": time.time() - t_start, "feature_names": names},
           "sigma": sigma, "fold_plans": {f"S{s}": {str(rt): {m: list(v[m]) for m in METHODS} for rt, v in fold_plans[s]["plans"].items()} for s in subjects},
           "fold_curves": {f"S{s}": {"Dfeat": fold_plans[s]["Dfeat"], "Dclf": fold_plans[s]["Dclf"], "bits_per_sample": fold_plans[s]["R_bits_per_sample"]} for s in subjects},
           "results": {}}
    for rt in targets:
        out["results"][str(rt)] = {}
        per = {m: {s: [] for s in subjects} for m in METHODS}
        bits = {m: {s: None for s in subjects} for m in METHODS}
        f1 = {m: {s: [] for s in subjects} for m in METHODS}
        for r in res:
            if r["rt"] == rt:
                per[r["m"]][r["sid"]].append(r["acc"]); f1[r["m"]][r["sid"]].append(r["f1"]); bits[r["m"]][r["sid"]] = (r["bits"], r["raw"], r["combo"])
        for m in METHODS:
            a = np.array([np.mean(per[m][s]) for s in subjects]); ff = np.array([np.mean(f1[m][s]) for s in subjects])
            ratio = sum(bits[m][s][1] for s in subjects) / sum(bits[m][s][0] for s in subjects)
            mean_lv = [float(np.mean([levels[bits[m][s][2][i]] for s in subjects])) for i in range(len(GROUPS))]
            med_lv = [float(np.median([levels[bits[m][s][2][i]] for s in subjects])) for i in range(len(GROUPS))]
            out["results"][str(rt)][m] = {"acc_mean": float(a.mean()), "acc_ci95": W.ci95(a), "f1_mean": float(ff.mean()), "ratio": ratio,
                                          "per_subject_acc": {f"S{s}": float(x) for s, x in zip(subjects, a)},
                                          "per_subject_ratio": {f"S{s}": bits[m][s][1] / bits[m][s][0] for s in subjects},
                                          "mean_level": dict(zip(GROUPS, mean_lv)), "median_level": dict(zip(GROUPS, med_lv))}
        au = np.array([out["results"][str(rt)]["uniform"]["per_subject_acc"][f"S{s}"] for s in subjects])
        for m in ("feat", "clf"):
            am = np.array([out["results"][str(rt)][m]["per_subject_acc"][f"S{s}"] for s in subjects])
            out["results"][str(rt)][m]["vs_uniform"] = W.paired_stats(am, au)
    # ---- Pareto summary: mean accuracy over targets, and paired test pooled over targets (subject x target)
    for m in ("feat", "clf"):
        d = [out["results"][str(rt)][m]["per_subject_acc"][f"S{s}"] - out["results"][str(rt)]["uniform"]["per_subject_acc"][f"S{s}"] for rt in targets for s in subjects]
        out.setdefault("pooled", {})[m] = {"mean_diff": float(np.mean(d)), "n": len(d),
                                            "p_wilcoxon": float(stats.wilcoxon(d).pvalue) if np.any(np.array(d) != 0) else 1.0}
    # ---- matched-ratio comparison: per subject, interpolate accuracy vs log(ratio) at common ratios
    grid = [8, 10, 12, 14, 16, 18, 20, 22, 24]
    out["matched"] = {"grid": grid, "per_method": {}}
    curves = {}
    for m in METHODS:
        curves[m] = {}
        for s in subjects:
            pts = sorted({(out["results"][str(rt)][m]["per_subject_ratio"][f"S{s}"], out["results"][str(rt)][m]["per_subject_acc"][f"S{s}"]) for rt in targets})
            xs = np.log([p[0] for p in pts]); ys = np.array([p[1] for p in pts])
            curves[m][s] = np.interp(np.log(grid), xs, ys, left=np.nan, right=np.nan)   # nan outside the achieved range
    for m in METHODS:
        A = np.array([curves[m][s] for s in subjects])   # subjects x grid
        out["matched"]["per_method"][m] = {"acc_at_ratio": {str(g): float(np.nanmean(A[:, i])) for i, g in enumerate(grid)},
                                           "n_valid": {str(g): int(np.sum(~np.isnan(A[:, i]))) for i, g in enumerate(grid)},
                                           "auc_per_subject": {f"S{s}": float(np.nanmean(curves[m][s])) for s in subjects}}
    for m in ("feat", "clf"):
        au = np.array([out["matched"]["per_method"]["uniform"]["auc_per_subject"][f"S{s}"] for s in subjects])
        am = np.array([out["matched"]["per_method"][m]["auc_per_subject"][f"S{s}"] for s in subjects])
        ok = ~np.isnan(au) & ~np.isnan(am)
        out["matched"]["per_method"][m]["auc_vs_uniform"] = W.paired_stats(am[ok], au[ok]) if ok.sum() > 1 else {}
        out["matched"]["per_method"][m]["acc_at_ratio_vs_uniform"] = {}
        for i, g in enumerate(grid):
            a1 = np.array([curves[m][s][i] for s in subjects]); a0 = np.array([curves["uniform"][s][i] for s in subjects]); ok = ~np.isnan(a1) & ~np.isnan(a0)
            out["matched"]["per_method"][m]["acc_at_ratio_vs_uniform"][str(g)] = {**(W.paired_stats(a1[ok], a0[ok]) if ok.sum() > 1 else {}), "n": int(ok.sum())}
    with open(os.path.join(args.out, "results_alloc.json"), "w") as f:
        json.dump(out, f, indent=1)
    write_outputs(out, args.out, args.main_results, P)
    P(f"total {time.time()-t_start:.0f}s")

def pfmt(p):
    return "$<$0.01" if p < 0.005 else f"{p:.2f}"

def write_outputs(out, od, main_results, P):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    R = out["results"]; targets = out["meta"]["targets"]; G = out["meta"]["groups"]
    # table
    rows = ["\\begin{tabular}{c" + "cc" * 3 + "cc}\n\\toprule\n",
            "Target & \\multicolumn{2}{c}{Uniform $\\epsilon$} & \\multicolumn{2}{c}{Feature-distortion alloc.} & \\multicolumn{2}{c}{Classifier-aware alloc.} & \\multicolumn{2}{c}{Clf.-aware vs.\\ uniform} \\\\\n",
            "$\\rho_t$ & $\\rho$ & Acc. (\\%) & $\\rho$ & Acc. (\\%) & $\\rho$ & Acc. (\\%) & $\\Delta$ & $p$ \\\\\n\\midrule\n"]
    for rt in targets:
        r = R[str(rt)]
        rows.append(f"$\\ge{rt}$ & {r['uniform']['ratio']:.1f}$\\times$ & {100*r['uniform']['acc_mean']:.1f} & {r['feat']['ratio']:.1f}$\\times$ & {100*r['feat']['acc_mean']:.1f} & "
                    f"{r['clf']['ratio']:.1f}$\\times$ & {100*r['clf']['acc_mean']:.1f} & {100*r['clf']['vs_uniform']['mean_diff']:+.1f} & {pfmt(r['clf']['vs_uniform']['p_wilcoxon'])} \\\\\n")
    rows.append("\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(od, "table_alloc.tex"), "w").writelines(rows)
    # allocation table: median chosen eps/sigma per group for clf method
    rows = ["\\begin{tabular}{c" + "c" * len(G) + "}\n\\toprule\n", "Target $\\rho_t$ & " + " & ".join(G) + " \\\\\n\\midrule\n"]
    for rt in targets:
        rows.append(f"$\\ge{rt}$ & " + " & ".join(f"{R[str(rt)]['clf']['median_level'][g]:g}" for g in G) + " \\\\\n")
    rows.append("\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(od, "table_alloc_levels.tex"), "w").writelines(rows)
    # matched-ratio table
    Mm = out["matched"]; grid = Mm["grid"]
    rows = ["\\begin{tabular}{l" + "c" * len(grid) + "}\n\\toprule\n", "Method & " + " & ".join(f"$\\rho={g}$" for g in grid) + " \\\\\n\\midrule\n"]
    lab = {"uniform": "Uniform $\\epsilon$", "feat": "Feature-distortion alloc.", "clf": "Classifier-aware alloc."}
    for m in ("uniform", "feat", "clf"):
        rows.append(lab[m] + " & " + " & ".join(f"{100*Mm['per_method'][m]['acc_at_ratio'][str(g)]:.1f}" for g in grid) + " \\\\\n")
    for m in ("feat", "clf"):
        rows.append(f"\\quad $p$ ({lab[m].split()[0]} vs.\\ uniform) & " + " & ".join((pfmt(Mm['per_method'][m]['acc_at_ratio_vs_uniform'][str(g)]['p_wilcoxon']) if 'p_wilcoxon' in Mm['per_method'][m]['acc_at_ratio_vs_uniform'][str(g)] else "---") for g in grid) + " \\\\\n")
    rows.append("\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(od, "table_alloc_matched.tex"), "w").writelines(rows)
    # macros
    L = []
    pre = {"dalia": "dalia", "nurse": "nurse"}.get(out["meta"].get("dataset"), "")
    _D = str.maketrans("0123456789", "ZABCDEFGHI")
    def macro(n, v): L.append(f"\\newcommand{{\\{pre}{str(n).translate(_D)}}}{{{v}}}\n")
    best_gain = max(targets, key=lambda rt: R[str(rt)]["clf"]["vs_uniform"]["mean_diff"])
    macro("allocNCombos", out["meta"]["n_combos"]); macro("allocNLevels", len(out["meta"]["levels"]))
    macro("allocBestTarget", best_gain); macro("allocBestGain", f"{100*R[str(best_gain)]['clf']['vs_uniform']['mean_diff']:+.1f}")
    macro("allocBestP", f"{R[str(best_gain)]['clf']['vs_uniform']['p_wilcoxon']:.3f}")
    macro("allocPooledDiff", f"{100*out['pooled']['clf']['mean_diff']:+.1f}"); macro("allocPooledP", f"{out['pooled']['clf']['p_wilcoxon']:.3f}")
    macro("allocPooledDiffFeat", f"{100*out['pooled']['feat']['mean_diff']:+.1f}"); macro("allocPooledPFeat", f"{out['pooled']['feat']['p_wilcoxon']:.3f}")
    for rt in targets:
        r = R[str(rt)]
        for m in ("uniform", "feat", "clf"):
            macro(f"alloc{m.capitalize()}Acc{rt}", f"{100*r[m]['acc_mean']:.1f}"); macro(f"alloc{m.capitalize()}Ratio{rt}", f"{r[m]['ratio']:.1f}")
        macro(f"allocClfSaving{rt}", f"{100*(1-1/r['clf']['ratio']):.1f}"); macro(f"allocClfDiff{rt}", f"{100*r['clf']['vs_uniform']['mean_diff']:+.1f}"); macro(f"allocClfP{rt}", pfmt(r['clf']['vs_uniform']['p_wilcoxon']))
    for m in ("feat", "clf"):
        v = Mm["per_method"][m].get("auc_vs_uniform", {})
        macro(f"allocAUCDiff{m.capitalize()}", f"{100*v.get('mean_diff', float('nan')):+.1f}"); macro(f"allocAUCP{m.capitalize()}", f"{v.get('p_wilcoxon', float('nan')):.3f}")
        for g in grid:
            macro(f"allocMatched{m.capitalize()}{g}", f"{100*Mm['per_method'][m]['acc_at_ratio'][str(g)]:.1f}")
            vv = Mm["per_method"][m]["acc_at_ratio_vs_uniform"][str(g)]
            macro(f"allocMatchedDiff{m.capitalize()}{g}", f"{100*vv.get('mean_diff', float('nan')):+.1f}"); macro(f"allocMatchedP{m.capitalize()}{g}", pfmt(vv.get('p_wilcoxon', 1.0)))
    for g in grid:
        macro(f"allocMatchedUniform{g}", f"{100*Mm['per_method']['uniform']['acc_at_ratio'][str(g)]:.1f}")
    macro("allocAUCUniform", f"{100*np.nanmean(list(Mm['per_method']['uniform']['auc_per_subject'].values())):.1f}")
    macro("allocAUCClf", f"{100*np.nanmean(list(Mm['per_method']['clf']['auc_per_subject'].values())):.1f}")
    macro("allocAUCFeat", f"{100*np.nanmean(list(Mm['per_method']['feat']['auc_per_subject'].values())):.1f}")
    open(os.path.join(od, "macros_alloc.tex"), "w").writelines(L)
    # figure: accuracy vs achieved ratio
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    sty = {"uniform": ("Uniform $\\epsilon$ (all groups)", "o-", "0.45"), "feat": ("Feature-distortion allocation", "s--", "tab:blue"), "clf": ("Classifier-aware allocation", "^-", "tab:red")}
    for m in ("uniform", "feat", "clf"):
        xs = [R[str(rt)][m]["ratio"] for rt in targets]; ys = [100 * R[str(rt)][m]["acc_mean"] for rt in targets]
        ax.plot(xs, ys, sty[m][1], color=sty[m][2], label=sty[m][0], ms=4, lw=1.2)
    try:
        M = json.load(open(main_results))
        raw = 100 * M["classification_matched"]["raw"]["acc"]["mean"]
        ax.axhline(raw, color="k", lw=0.8, ls=":", label=f"Uncompressed ({raw:.1f}%)")
        ax.set_ylim(bottom=max(ax.get_ylim()[0], raw - 20))
        for n, c in M["compression"].items():
            if c["method"] == "downsample":
                ax.plot(c["ratio"], 100 * M["classification_matched"][n]["acc"]["mean"], "x", color="tab:green", ms=6, label="Downsampling" if c["param"] == 2 else None)
            if c["method"] == "zfp":
                ax.plot(c["ratio"], 100 * M["classification_matched"][n]["acc"]["mean"], "+", color="tab:purple", ms=6, label="ZFP" if c["param"] == M["meta"]["eps_grid"][0] else None)
    except Exception as e:
        P("main results not found for figure:", e)
    ax.set_xlabel("Rate-weighted compression ratio $\\rho$"); ax.set_ylabel("LOSO accuracy (%)"); ax.grid(alpha=0.3); ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout(); fig.savefig(os.path.join(od, "fig_pareto.pdf")); plt.close(fig)
    # figure: distortion curves (mean over folds) per group
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 2.1))
    plt.rcParams.update({"font.size": 7})
    lv = out["meta"]["levels"][1:]
    for g in G:
        dc = np.mean([out["fold_curves"][s]["Dclf"][g][1:] for s in out["fold_curves"]], axis=0)
        bps = np.mean([out["fold_curves"][s]["bits_per_sample"][g][1:] for s in out["fold_curves"]], axis=0)
        axes[0].plot(lv, dc, "o-", ms=3, label=g); axes[1].plot(lv, bps, "o-", ms=3, label=g)
    for a in axes:
        a.set_xscale("log"); a.set_xlabel("$\\epsilon/\\sigma_g$", fontsize=7); a.grid(alpha=0.3); a.tick_params(labelsize=6)
    axes[0].set_yscale("log"); axes[0].set_ylabel("$D_g^{\\mathrm{clf}}$ (log-prob. MSE)", fontsize=7); axes[0].legend(fontsize=6)
    axes[1].set_ylabel("$R_g$ (bits/sample)", fontsize=7)
    fig.tight_layout(); fig.savefig(os.path.join(od, "fig_curves.pdf")); plt.close(fig)
    # summary text
    S = [f"ALLOCATION RESULTS ({len(out['meta']['subjects'])} subjects, {out['meta']['seeds']} seeds, {out['meta']['n_combos']} combos)"]
    S.append(f"{'target':>7}{'uni_ratio':>10}{'uni_acc':>8}{'feat_ratio':>11}{'feat_acc':>9}{'clf_ratio':>10}{'clf_acc':>8}{'clf-uni':>8}{'p':>7}   clf median eps/sigma per group " + str(G))
    for rt in targets:
        r = R[str(rt)]
        S.append(f"{rt:>7}{r['uniform']['ratio']:>10.1f}{100*r['uniform']['acc_mean']:>8.1f}{r['feat']['ratio']:>11.1f}{100*r['feat']['acc_mean']:>9.1f}{r['clf']['ratio']:>10.1f}{100*r['clf']['acc_mean']:>8.1f}"
                 f"{100*r['clf']['vs_uniform']['mean_diff']:>+8.1f}{r['clf']['vs_uniform']['p_wilcoxon']:>7.2f}   {[r['clf']['median_level'][g] for g in G]}")
    S.append(f"pooled clf-uniform diff {100*out['pooled']['clf']['mean_diff']:+.2f} (p={out['pooled']['clf']['p_wilcoxon']:.4f}, n={out['pooled']['clf']['n']}); feat-uniform {100*out['pooled']['feat']['mean_diff']:+.2f} (p={out['pooled']['feat']['p_wilcoxon']:.4f})")
    S.append("Matched-ratio accuracy (interpolated per subject): ratio -> uniform / feat / clf  [p clf vs uniform]")
    for g in grid:
        pm = Mm["per_method"]; pv = pm["clf"]["acc_at_ratio_vs_uniform"][str(g)]
        S.append(f"  rho={g:>3}: {100*pm['uniform']['acc_at_ratio'][str(g)]:.1f} / {100*pm['feat']['acc_at_ratio'][str(g)]:.1f} / {100*pm['clf']['acc_at_ratio'][str(g)]:.1f}  [p={pv.get('p_wilcoxon', float('nan')):.3f}, n={pv['n']}]")
    for m in ("feat", "clf"):
        v = Mm["per_method"][m].get("auc_vs_uniform", {})
        S.append(f"  mean accuracy over rho in [{grid[0]},{grid[-1]}]: {m} - uniform = {100*v.get('mean_diff', float('nan')):+.2f} (p={v.get('p_wilcoxon', float('nan')):.4f})")
    txt = "\n".join(S); open(os.path.join(od, "summary_alloc.txt"), "w").write(txt + "\n"); P("\n" + txt)

if __name__ == "__main__":
    main()
