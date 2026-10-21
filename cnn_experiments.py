#!/usr/bin/env python
"""
Stronger model check: a 1-D CNN trained directly on reconstructed waveforms (no hand-crafted features).
All six wrist channels are brought to 32 Hz (BVP: mean of sample pairs; EDA/TEMP: sample-and-hold), giving a
6 x 1920 tensor per 60 s window. Each channel is z-scored within the window (removes subject-specific offsets),
and the 12 per-channel window means and standard deviations, standardized with training-fold statistics, are
fed to the classifier head as side inputs so that level information is not lost.
LOSO, several seeds, matched (train+test reconstructed) and mismatched (train uncompressed) scenarios.
Configs: raw, proposed at selected eps, downsampling D=4 and D=8.

Usage: python cnn_experiments.py --wesad_dir data/WESAD --out results_cnn [--quick]
"""
import argparse, json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import wesad_compression_experiments as W
import torch, torch.nn as nn

EPS_CNN = [0.01, 0.02, 0.05, 0.1, 0.2]
FS_OUT = 32

def to32(ch):
    bvp = ch["BVP"][: len(ch["BVP"]) // 2 * 2].reshape(-1, 2).mean(axis=1)
    rep = lambda x: np.repeat(x, 8)
    n = min(len(bvp), len(rep(ch["EDA"])), len(ch["ACC_x"]))
    return np.stack([bvp[:n], rep(ch["EDA"])[:n], rep(ch["TEMP"])[:n], ch["ACC_x"][:n], ch["ACC_y"][:n], ch["ACC_z"][:n]]).astype(np.float32)

def windows(sig32, label, dur, win=W.WIN_SEC, step=W.STEP_SEC):
    X, y = [], []
    for t0 in range(0, dur - win + 1, step):
        lw = label[t0 * W.LABEL_FS:(t0 + win) * W.LABEL_FS]
        vals, cnts = np.unique(lw, return_counts=True); maj = int(vals[np.argmax(cnts)]); purity = cnts.max() / len(lw)
        if maj not in W.LABELS_KEEP or purity < W.MIN_PURITY:
            continue
        seg = sig32[:, t0 * FS_OUT:(t0 + win) * FS_OUT]
        if seg.shape[1] == win * FS_OUT:
            X.append(seg); y.append(1 if maj == 2 else 0)
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.int64)

class Net(nn.Module):
    def __init__(self, cin=6, nside=12, ncls=2):
        super().__init__()
        def blk(i, o, k, s): return nn.Sequential(nn.Conv1d(i, o, k, stride=s, padding=k // 2), nn.BatchNorm1d(o), nn.ReLU())
        self.f = nn.Sequential(blk(cin, 16, 7, 2), blk(16, 32, 7, 2), blk(32, 64, 5, 2), blk(64, 64, 5, 2), nn.AdaptiveAvgPool1d(1), nn.Flatten())
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(64 + nside, 32), nn.ReLU(), nn.Linear(32, ncls))
    def forward(self, x, s): return self.head(torch.cat([self.f(x), s], dim=1))

def prep(X):
    """per-window per-channel z-score + side statistics (mean, std per channel)."""
    mu = X.mean(axis=2, keepdims=True); sd = X.std(axis=2, keepdims=True) + 1e-6
    return ((X - mu) / sd).astype(np.float32), np.concatenate([mu[:, :, 0], sd[:, :, 0]], axis=1).astype(np.float32)

def train_eval(Xtr, ytr, Xte_list, seed, device, epochs=20, bs=64):
    torch.manual_seed(seed); np.random.seed(seed)
    Xz, S = prep(Xtr); smu = S.mean(axis=0, keepdims=True); ssd = S.std(axis=0, keepdims=True) + 1e-6
    Xt = torch.tensor(Xz); St = torch.tensor((S - smu) / ssd); yt = torch.tensor(ytr)
    w = torch.tensor([len(ytr) / (2 * max((ytr == c).sum(), 1)) for c in (0, 1)], dtype=torch.float32)
    net = Net().to(device); opt = torch.optim.Adam(net.parameters(), lr=1e-3, weight_decay=1e-4); lossf = nn.CrossEntropyLoss(weight=w.to(device))
    n = len(yt)
    for ep in range(epochs):
        net.train(); perm = torch.randperm(n)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]; xb = Xt[idx].to(device); sb = St[idx].to(device); yb = yt[idx].to(device)
            opt.zero_grad(); loss = lossf(net(xb, sb), yb); loss.backward(); opt.step()
    net.eval(); outs = []
    with torch.no_grad():
        for Xte in Xte_list:
            Xz, S = prep(Xte); xb = torch.tensor(Xz).to(device); sb = torch.tensor((S - smu) / ssd).to(device); preds = []
            for i in range(0, len(xb), 256):
                preds.append(net(xb[i:i + 256], sb[i:i + 256]).argmax(1).cpu().numpy())
            outs.append(np.concatenate(preds) if preds else np.array([], dtype=int))
    return outs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad_dir", required=True); ap.add_argument("--out", default="results_cnn")
    ap.add_argument("--seeds", type=int, default=3); ap.add_argument("--epochs", type=int, default=30); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--rewrite_tex", action="store_true")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True); t0 = time.time()
    if args.rewrite_tex:
        write_outputs(json.load(open(os.path.join(args.out, "results.json"))), args.out, print); return
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    subjects = W.SUBJECTS_ALL[:3] if args.quick else W.SUBJECTS_ALL
    eps_list = [0.05] if args.quick else EPS_CNN
    seeds = 1 if args.quick else args.seeds; epochs = 3 if args.quick else args.epochs
    logf = open(os.path.join(args.out, "run.log"), "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); logf.write(s + "\n"); logf.flush()
    P(f"device={device} subjects={subjects} eps={eps_list} seeds={seeds} epochs={epochs}")
    data = {}; acc = {g: [0.0, 0.0, 0] for g in W.FS}
    for sid in subjects:
        ch, label, dur = W.load_subject(args.wesad_dir, sid); data[sid] = (ch, label, dur)
        for c in W.CHANNELS:
            g = W.CH_GROUP[c]; v = ch[c]; acc[g][0] += v.sum(); acc[g][1] += (v ** 2).sum(); acc[g][2] += len(v)
    sigma = {g: math.sqrt(max(acc[g][1] / acc[g][2] - (acc[g][0] / acc[g][2]) ** 2, 0.0)) for g in W.FS}
    configs = [("raw", None)] + [("proposed", e) for e in eps_list] + [("downsample", 4), ("downsample", 8)]
    cname = lambda m, p: m if p is None else f"{m}_{p:g}"
    Xs = {}; comp = {}
    for m, p in configs:
        name = cname(m, p); Xs[name] = {}; comp[name] = {"method": m, "param": p, "bits": 0.0, "raw_bits": 0.0}
        for sid in subjects:
            ch, label, dur = data[sid]; rec = {}
            for c in W.CHANNELS:
                x = ch[c]; fs = W.FS[W.CH_GROUP[c]]
                if m == "raw":
                    xh, bits = x, W.RAW_BITS * len(x)
                else:
                    eps = p * sigma[W.CH_GROUP[c]] if m == "proposed" else p
                    xh, bits, err, _ = W.compress_stream(x, m, eps, fs)
                    if m == "proposed":
                        assert err <= eps * (1 + 1e-9)
                rec[c] = xh; comp[name]["bits"] += bits; comp[name]["raw_bits"] += W.RAW_BITS * len(x)
            Xs[name][sid] = windows(to32(rec), label, dur)
        comp[name]["ratio"] = comp[name]["raw_bits"] / comp[name]["bits"]
        P(f"[{name}] ratio={comp[name]['ratio']:.2f} windows={sum(len(Xs[name][s][1]) for s in subjects)} ({time.time()-t0:.0f}s)")
    names = list(comp.keys())
    per = {n: {s: {"acc": [], "f1": []} for s in subjects} for n in names}; per_mm = {n: {s: {"acc": [], "f1": []} for s in subjects} for n in names}
    for sid in subjects:
        for seed in range(seeds):
            for name in names:
                Xtr = np.concatenate([Xs[name][s][0] for s in subjects if s != sid]); ytr = np.concatenate([Xs[name][s][1] for s in subjects if s != sid])
                if name == "raw":
                    preds = train_eval(Xtr, ytr, [Xs[n][sid][0] for n in names], seed, device, epochs)
                    for i, n in enumerate(names):
                        mm = W.metrics(Xs[n][sid][1], preds[i]); per_mm[n][sid]["acc"].append(mm["acc"]); per_mm[n][sid]["f1"].append(mm["f1"])
                    mt = W.metrics(Xs[name][sid][1], preds[0])
                else:
                    mt = W.metrics(Xs[name][sid][1], train_eval(Xtr, ytr, [Xs[name][sid][0]], seed, device, epochs)[0])
                per[name][sid]["acc"].append(mt["acc"]); per[name][sid]["f1"].append(mt["f1"])
            P(f"S{sid} seed{seed}: " + " ".join(f"{n}={100*per[n][sid]['acc'][-1]:.1f}" for n in names) + f" ({time.time()-t0:.0f}s)")
    def summarize(tab):
        out = {}; ref = {k: np.array([np.mean(tab["raw"][s][k]) for s in subjects]) for k in ("acc", "f1")}
        for n in names:
            out[n] = {}
            for k in ("acc", "f1"):
                v = np.array([np.mean(tab[n][s][k]) for s in subjects])
                out[n][k] = {"mean": float(v.mean()), "ci95": W.ci95(v), "per_subject": {f"S{s}": float(x) for s, x in zip(subjects, v)}, "vs_raw": W.paired_stats(v, ref[k])}
        return out
    R = {"meta": {"subjects": subjects, "seeds": seeds, "epochs": epochs, "device": str(device), "eps": eps_list, "runtime_s": time.time() - t0,
                  "n_params": sum(p.numel() for p in Net().parameters())}, "compression": comp, "classification_matched": summarize(per), "classification_mismatched": summarize(per_mm)}
    json.dump(R, open(os.path.join(args.out, "results.json"), "w"), indent=1)
    write_outputs(R, args.out, P)

def write_outputs(R, out, P):
    comp = R["compression"]; subjects = R["meta"]["subjects"]; seeds = R["meta"]["seeds"]; epochs = R["meta"]["epochs"]
    M, U = R["classification_matched"], R["classification_mismatched"]
    key = {0.005: "Ea", 0.01: "Eb", 0.02: "Ec", 0.05: "Ed", 0.1: "Ee", 0.2: "Ef"}
    L = [f"\\newcommand{{\\cnnRawAcc}}{{{100*M['raw']['acc']['mean']:.1f}}}\n", f"\\newcommand{{\\cnnRawAccCI}}{{{100*M['raw']['acc']['ci95']:.1f}}}\n", f"\\newcommand{{\\cnnRawF}}{{{100*M['raw']['f1']['mean']:.1f}}}\n",
         f"\\newcommand{{\\cnnNParams}}{{{R['meta']['n_params']}}}\n", f"\\newcommand{{\\cnnNSeeds}}{{{seeds}}}\n", f"\\newcommand{{\\cnnEpochs}}{{{epochs}}}\n"]
    rows = ["\\begin{tabular}{lcccccc}\n\\toprule\n", "Config. & Ratio $\\rho$ & Acc. (\\%) & $\\Delta$Acc. & $p$ & F1 (\\%) & Acc. mismatched (\\%) \\\\\n\\midrule\n"]
    for n, c in comp.items():
        lab = "Uncompressed" if c["method"] == "raw" else (f"Proposed $\\epsilon={c['param']:g}\\sigma$" if c["method"] == "proposed" else f"Downsampling $D={c['param']}$")
        d = "---" if n == "raw" else f"{100*M[n]['acc']['vs_raw']['mean_diff']:+.1f}"; pp = "---" if n == "raw" else f"{M[n]['acc']['vs_raw']['p_wilcoxon']:.2f}"
        rows.append(f"{lab} & {c['ratio']:.1f}$\\times$ & {100*M[n]['acc']['mean']:.1f} $\\pm$ {100*M[n]['acc']['ci95']:.1f} & {d} & {pp} & {100*M[n]['f1']['mean']:.1f} & {100*U[n]['acc']['mean']:.1f} \\\\\n")
        if c["method"] == "proposed":
            k = key[c["param"]]; L += [f"\\newcommand{{\\cnnAcc{k}}}{{{100*M[n]['acc']['mean']:.1f}}}\n", f"\\newcommand{{\\cnnDAcc{k}}}{{{100*M[n]['acc']['vs_raw']['mean_diff']:+.1f}}}\n", f"\\newcommand{{\\cnnP{k}}}{{{M[n]['acc']['vs_raw']['p_wilcoxon']:.2f}}}\n", f"\\newcommand{{\\cnnAccMM{k}}}{{{100*U[n]['acc']['mean']:.1f}}}\n"]
        if c["method"] == "downsample":
            L += [f"\\newcommand{{\\cnnDsAcc{'ABCDEFGHI'[c['param']-1]}}}{{{100*M[n]['acc']['mean']:.1f}}}\n", f"\\newcommand{{\\cnnDsDAcc{'ABCDEFGHI'[c['param']-1]}}}{{{100*M[n]['acc']['vs_raw']['mean_diff']:+.1f}}}\n", f"\\newcommand{{\\cnnDsAccMM{'ABCDEFGHI'[c['param']-1]}}}{{{100*U[n]['acc']['mean']:.1f}}}\n"]
    rows.append("\\bottomrule\n\\end{tabular}\n")
    open(os.path.join(out, "table_cnn.tex"), "w").writelines(rows); open(os.path.join(out, "macros_cnn.tex"), "w").writelines(L)
    S = [f"CNN results ({len(subjects)} subjects, {seeds} seeds, {epochs} epochs, {R['meta']['n_params']} params, {R['meta']['runtime_s']/60:.1f} min)", f"{'config':<16}{'ratio':>7}{'acc':>7}{'dAcc':>7}{'p':>6}{'F1':>7}{'acc_mm':>8}"]
    for n, c in comp.items():
        S.append(f"{n:<16}{c['ratio']:>7.2f}{100*M[n]['acc']['mean']:>7.1f}{100*M[n]['acc']['vs_raw']['mean_diff']:>+7.1f}{M[n]['acc']['vs_raw']['p_wilcoxon']:>6.2f}{100*M[n]['f1']['mean']:>7.1f}{100*U[n]['acc']['mean']:>8.1f}")
    open(os.path.join(out, "summary.txt"), "w").write("\n".join(S) + "\n"); P("\n" + "\n".join(S))

if __name__ == "__main__":
    main()
