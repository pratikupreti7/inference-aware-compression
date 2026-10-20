#!/usr/bin/env python
"""
Inference-preserving error-bounded compression experiments on WESAD (Empatica E4 wrist data).

Pipeline
--------
1. Load WESAD/S*/S*.pkl, keep wrist channels BVP (64 Hz), EDA (4 Hz), TEMP (4 Hz), ACC x/y/z (32 Hz).
2. Compress every channel of every subject with several methods / settings:
     proposed : closed-loop 1-D Lorenzo prediction on *reconstructed* samples + uniform quantization
                (bin width 2*eps) + per-window Huffman coding (table overhead counted).
     uniform  : scalar quantization (bin width 2*eps), no prediction, best of bit-packing / Huffman.
     zfp      : ZFP fixed-accuracy mode (tolerance = eps), whole per-subject stream (favourable to ZFP).
     downsample: keep every D-th sample (float32), reconstruct by linear interpolation. Not error-bounded.
     lossless : zlib level 9 on float32 bytes.
     openloop : the *incorrect* predictor used in the earlier draft (predicts from original samples).
                Diagnostic only - shows that the error bound is violated.
   The absolute error bound of a channel is eps_abs = eps_rel * sigma_channel, where sigma_channel
   is the global standard deviation of that channel over the whole corpus (a fixed device constant).
3. Verify max |x_hat - x| <= eps_abs for every error-bounded method, subject and channel, and that an
   independent decoder reproduces the encoder's reconstruction bit-exactly.
4. Extract 60 s window features from the reconstructed signals, train a feed-forward neural network
   (sklearn MLPClassifier) for binary stress detection (stress vs. baseline+amusement) under
   leave-one-subject-out cross-validation, several random seeds.
   Two scenarios: matched (train and test on reconstructed data) and mismatched (train on
   uncompressed, test on reconstructed).
5. Paired statistics across subjects vs. the uncompressed baseline (Wilcoxon signed-rank, paired t).
6. Write results/results.json, results/macros.tex, results/table_*.tex, results/verification.txt,
   results/summary.txt.  paper.tex reads the .tex files.

Usage: python wesad_compression_experiments.py --wesad_dir WESAD --out results [--quick]
"""
import argparse, heapq, json, math, os, pickle, sys, time, zlib, platform
from collections import Counter
import numpy as np
from scipy import stats, signal

try:
    from numba import njit
    HAVE_NUMBA = True
except Exception:  # pragma: no cover
    HAVE_NUMBA = False
    def njit(*a, **k):
        def deco(f):
            return f
        return deco if not a or not callable(a[0]) else a[0]

try:
    import zfpy
    HAVE_ZFP = True
except Exception:
    HAVE_ZFP = False

# ----------------------------------------------------------------------------- constants
FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
CHANNELS = ["BVP", "EDA", "TEMP", "ACC_x", "ACC_y", "ACC_z"]
CH_GROUP = {"BVP": "BVP", "EDA": "EDA", "TEMP": "TEMP", "ACC_x": "ACC", "ACC_y": "ACC", "ACC_z": "ACC"}
LABEL_FS = 700
SUBJECTS_ALL = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17]
EPS_GRID = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
DS_GRID = [2, 4, 8]
WIN_SEC, STEP_SEC = 60, 10          # feature windows
TX_WIN_SEC = 60                     # device transmit / compression window
KMAX = 32767                        # max |quantization index| before escape
RAW_BITS = 32                       # uncompressed representation: IEEE float32 per sample
# Native sensor resolution of the E4 channels as stored in WESAD (BVP/TEMP on a 0.01 grid, ACC 8-bit integers,
# EDA not on a grid). eps = resolution/2 makes the proposed codec exact for BVP/TEMP/ACC ("native" setting).
NATIVE_EPS = {"BVP": 0.005, "TEMP": 0.005, "ACC": 0.5, "EDA": 3.45e-5}
HUFF_SYM_BITS, HUFF_LEN_BITS = 16, 6
LABELS_KEEP = {1: "baseline", 2: "stress", 3: "amusement"}
MIN_PURITY = 0.9

# ----------------------------------------------------------------------------- entropy coding
def huffman_code_lengths(counts):
    """Return dict symbol -> code length for the given Counter (canonical Huffman)."""
    if len(counts) == 1:
        return {next(iter(counts)): 1}
    heap = [(c, i, [s]) for i, (s, c) in enumerate(counts.items())]
    heapq.heapify(heap)
    lengths = {s: 0 for s in counts}
    uid = len(heap)
    while len(heap) > 1:
        c1, _, s1 = heapq.heappop(heap)
        c2, _, s2 = heapq.heappop(heap)
        for s in s1 + s2:
            lengths[s] += 1
        heapq.heappush(heap, (c1 + c2, uid, s1 + s2))
        uid += 1
    return lengths

def huffman_bits(symbols, sym_bits=HUFF_SYM_BITS):
    """Total bits to transmit `symbols` with a per-window Huffman table (table overhead included)."""
    counts = Counter(symbols.tolist())
    lengths = huffman_code_lengths(counts)
    data_bits = sum(counts[s] * lengths[s] for s in counts)
    table_bits = 16 + len(counts) * (sym_bits + HUFF_LEN_BITS)
    return data_bits + table_bits, len(counts)

# ----------------------------------------------------------------------------- proposed codec
@njit(cache=True)
def _lorenzo_encode(x, q, kmax):
    n = x.shape[0]
    xhat = np.empty(n)
    sym = np.empty(n, np.int64)
    esc = np.zeros(n, np.bool_)
    for i in range(n):
        if i < 2:
            k = int(math.floor(x[i] / q + 0.5))
            xhat[i] = k * q; sym[i] = k; esc[i] = True
        else:
            pred = 2.0 * xhat[i - 1] - xhat[i - 2]
            r = x[i] - pred
            k = int(math.floor(r / q + 0.5))
            if k > kmax or k < -kmax:
                k2 = int(math.floor(x[i] / q + 0.5))
                xhat[i] = k2 * q; sym[i] = k2; esc[i] = True
            else:
                xhat[i] = pred + k * q; sym[i] = k; esc[i] = False
    return xhat, sym, esc

@njit(cache=True)
def _lorenzo_decode(sym, esc, q):
    n = sym.shape[0]
    xhat = np.empty(n)
    for i in range(n):
        if esc[i] or i < 2:
            xhat[i] = sym[i] * q
        else:
            xhat[i] = 2.0 * xhat[i - 1] - xhat[i - 2] + sym[i] * q
    return xhat

@njit(cache=True)
def _openloop_encode(x, q):
    """Incorrect predictor from the earlier draft: predicts from ORIGINAL samples (encoder) but the
    decoder only has reconstructed samples -> errors accumulate. Diagnostic only."""
    n = x.shape[0]
    sym = np.empty(n, np.int64)
    xhat = np.empty(n)
    for i in range(n):
        if i < 2:
            k = int(math.floor(x[i] / q + 0.5)); sym[i] = k; xhat[i] = k * q
        else:
            pred_enc = 2.0 * x[i - 1] - x[i - 2]
            k = int(math.floor((x[i] - pred_enc) / q + 0.5)); sym[i] = k
            xhat[i] = 2.0 * xhat[i - 1] - xhat[i - 2] + k * q      # what the decoder can compute
    return xhat, sym

def encode_proposed_window(x, eps):
    q = 2.0 * eps
    xhat, sym, esc = _lorenzo_encode(x, q, KMAX)
    # bitstream: 16-bit length + 2 boundary int32 + escaped samples (ESC code + int32) + Huffman data
    n_esc = int(esc[2:].sum())
    coded = sym[2:].copy()
    coded[esc[2:]] = -KMAX - 1          # ESC symbol shares the Huffman alphabet
    bits, nsym = huffman_bits(coded)
    bits += 16 + 2 * 32 + n_esc * 32
    # independent decoder check
    xdec = _lorenzo_decode(sym, esc, q)
    assert np.array_equal(xdec, xhat), "decoder mismatch"
    return xhat, bits, {"n_esc": n_esc, "n_sym": nsym}

def encode_uniform_window(x, eps):
    q = 2.0 * eps
    k = np.floor(x / q + 0.5).astype(np.int64)
    xhat = k * q
    kmin, kmax = int(k.min()), int(k.max())
    nlev = kmax - kmin + 1
    packed = 2 * 32 + len(k) * max(1, math.ceil(math.log2(nlev))) if nlev > 1 else 2 * 32 + len(k)
    hbits, _ = huffman_bits(k - kmin, sym_bits=32)
    hbits += 32
    return xhat, 16 + 1 + min(packed, hbits), {}

def encode_downsample_window(x, D):
    n = len(x)
    idx = np.arange(0, n, D)
    kept = x[idx].astype(np.float32).astype(np.float64)
    xhat = np.interp(np.arange(n), idx, kept)
    return xhat, 16 + 32 * len(idx), {}

def encode_zfp_stream(x, eps):
    x32 = x.astype(np.float32)
    comp = zfpy.compress_numpy(x32, tolerance=float(eps))
    xhat = zfpy.decompress_numpy(comp).astype(np.float64)
    return xhat, 8 * len(comp), {}

def encode_lossless_stream(x):
    b = x.astype(np.float32).tobytes()
    c = zlib.compress(b, 9)
    return x.astype(np.float32).astype(np.float64), 8 * len(c), {}

def compress_stream(x, method, param, fs):
    """Compress a whole per-subject channel stream. Returns xhat, total bits, max abs err, extras."""
    n = len(x)
    if method in ("zfp", "lossless"):
        xhat, bits, ex = encode_zfp_stream(x, param) if method == "zfp" else encode_lossless_stream(x)
        if method == "lossless":
            # lossless w.r.t. float32 representation: error vs float64 original is float32 rounding
            pass
        return xhat, bits, float(np.max(np.abs(xhat - x))), ex
    if method == "openloop":
        xhat, _ = _openloop_encode(x, 2.0 * param)
        return xhat, float("nan"), float(np.max(np.abs(xhat - x))), {}
    w = TX_WIN_SEC * fs
    xhat = np.empty(n); bits = 0; n_esc = 0; n_sym = 0; nwin = 0
    for s in range(0, n, w):
        seg = x[s:s + w]
        if method == "proposed":
            xh, b, ex = encode_proposed_window(seg, param)
            n_esc += ex["n_esc"]; n_sym += ex["n_sym"]
        elif method == "uniform":
            xh, b, ex = encode_uniform_window(seg, param)
        elif method == "downsample":
            xh, b, ex = encode_downsample_window(seg, param)
        else:
            raise ValueError(method)
        xhat[s:s + w] = xh; bits += b; nwin += 1
    return xhat, bits, float(np.max(np.abs(xhat - x))), {"n_esc": n_esc, "mean_alphabet": n_sym / max(nwin, 1)}

# ----------------------------------------------------------------------------- data loading
def load_subject(wesad_dir, sid):
    p = os.path.join(wesad_dir, f"S{sid}", f"S{sid}.pkl")
    with open(p, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    w = d["signal"]["wrist"]
    ch = {
        "BVP": w["BVP"][:, 0].astype(np.float64),
        "EDA": w["EDA"][:, 0].astype(np.float64),
        "TEMP": w["TEMP"][:, 0].astype(np.float64),
        "ACC_x": w["ACC"][:, 0].astype(np.float64),
        "ACC_y": w["ACC"][:, 1].astype(np.float64),
        "ACC_z": w["ACC"][:, 2].astype(np.float64),
    }
    label = np.asarray(d["label"]).astype(np.int16)
    # common duration (seconds) across label stream and wrist channels
    dur = min([len(label) / LABEL_FS] + [len(ch[c]) / FS[CH_GROUP[c]] for c in ch])
    dur = int(math.floor(dur))
    for c in ch:
        ch[c] = ch[c][: dur * FS[CH_GROUP[c]]]
    label = label[: dur * LABEL_FS]
    return ch, label, dur

# ----------------------------------------------------------------------------- features
def _slope(x):
    t = np.arange(len(x), dtype=float)
    return float(np.polyfit(t, x, 1)[0]) if len(x) > 1 else 0.0

def feat_bvp(x, fs=64):
    f = {"bvp_mean": x.mean(), "bvp_std": x.std(), "bvp_min": x.min(), "bvp_max": x.max()}
    xs = x - x.mean()
    try:
        b, a = signal.butter(2, [0.5 / (fs / 2), 8.0 / (fs / 2)], btype="band")
        xf = signal.filtfilt(b, a, xs)
    except Exception:
        xf = xs
    peaks, _ = signal.find_peaks(xf, distance=int(0.33 * fs), prominence=0.5 * xf.std() if xf.std() > 0 else None)
    if len(peaks) >= 4:
        ibi = np.diff(peaks) / fs
        ibi = ibi[(ibi > 0.3) & (ibi < 2.0)]
    else:
        ibi = np.array([])
    if len(ibi) >= 3:
        hr = 60.0 / ibi
        dibi = np.diff(ibi)
        f.update({"hr_mean": hr.mean(), "hr_std": hr.std(), "ibi_mean": ibi.mean(), "sdnn": ibi.std(),
                  "rmssd": np.sqrt(np.mean(dibi ** 2)), "pnn50": np.mean(np.abs(dibi) > 0.05),
                  "n_beats": len(ibi)})
    else:
        f.update({"hr_mean": 0.0, "hr_std": 0.0, "ibi_mean": 0.0, "sdnn": 0.0, "rmssd": 0.0, "pnn50": 0.0, "n_beats": 0.0})
    fr, p = signal.welch(xs, fs=fs, nperseg=min(len(xs), 512))
    tot = p.sum() + 1e-12
    f["bvp_p_lf"] = p[(fr >= 0.5) & (fr < 1.5)].sum() / tot
    f["bvp_p_hf"] = p[(fr >= 1.5) & (fr < 4.0)].sum() / tot
    f["bvp_f_peak"] = fr[np.argmax(p)]
    return f

def feat_eda(x, fs=4):
    f = {"eda_mean": x.mean(), "eda_std": x.std(), "eda_min": x.min(), "eda_max": x.max(),
         "eda_range": x.max() - x.min(), "eda_slope": _slope(x)}
    try:
        b, a = signal.butter(1, 0.05 / (fs / 2), btype="low")
        tonic = signal.filtfilt(b, a, x)
    except Exception:
        tonic = np.full_like(x, x.mean())
    phasic = x - tonic
    f["scl_mean"] = tonic.mean(); f["scr_std"] = phasic.std()
    peaks, props = signal.find_peaks(phasic, prominence=0.01, distance=fs)
    f["scr_count"] = float(len(peaks))
    f["scr_amp_mean"] = float(props["prominences"].mean()) if len(peaks) else 0.0
    dx = np.diff(x) * fs
    f["eda_dmax"] = dx.max() if len(dx) else 0.0
    f["eda_dmean"] = np.abs(dx).mean() if len(dx) else 0.0
    return f

def feat_temp(x, fs=4):
    return {"temp_mean": x.mean(), "temp_std": x.std(), "temp_min": x.min(), "temp_max": x.max(),
            "temp_slope": _slope(x), "temp_range": x.max() - x.min()}

def feat_acc(ax, ay, az, fs=32):
    f = {}
    for n, v in (("x", ax), ("y", ay), ("z", az)):
        f[f"acc_{n}_mean"] = v.mean(); f[f"acc_{n}_std"] = v.std()
        f[f"acc_{n}_absint"] = np.abs(v - v.mean()).sum() / len(v)
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)
    f["acc_mag_mean"] = mag.mean(); f["acc_mag_std"] = mag.std(); f["acc_mag_max"] = mag.max()
    f["acc_jerk"] = np.abs(np.diff(mag)).mean() if len(mag) > 1 else 0.0
    fr, p = signal.welch(mag - mag.mean(), fs=fs, nperseg=min(len(mag), 256))
    f["acc_f_peak"] = fr[np.argmax(p)]
    f["acc_p_tot"] = np.log1p(p.sum())
    return f

def window_features(ch, label, dur, win=WIN_SEC, step=STEP_SEC):
    X, y, names, lab_full = [], [], None, []
    for t0 in range(0, dur - win + 1, step):
        lw = label[t0 * LABEL_FS:(t0 + win) * LABEL_FS]
        vals, cnts = np.unique(lw, return_counts=True)
        maj = int(vals[np.argmax(cnts)]); purity = cnts.max() / len(lw)
        if maj not in LABELS_KEEP or purity < MIN_PURITY:
            continue
        f = {}
        f.update(feat_bvp(ch["BVP"][t0 * 64:(t0 + win) * 64]))
        f.update(feat_eda(ch["EDA"][t0 * 4:(t0 + win) * 4]))
        f.update(feat_temp(ch["TEMP"][t0 * 4:(t0 + win) * 4]))
        f.update(feat_acc(ch["ACC_x"][t0 * 32:(t0 + win) * 32], ch["ACC_y"][t0 * 32:(t0 + win) * 32], ch["ACC_z"][t0 * 32:(t0 + win) * 32]))
        if names is None:
            names = list(f.keys())
        X.append([f[k] for k in names]); y.append(1 if maj == 2 else 0); lab_full.append(maj)
    X = np.nan_to_num(np.array(X, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    return X, np.array(y, dtype=int), np.array(lab_full), names

# ----------------------------------------------------------------------------- classifier
def fit_eval(Xtr, ytr, Xte_list, seed, hidden=(64, 32)):
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    rng = np.random.RandomState(seed)
    # balance classes by random oversampling of the minority class
    idx1 = np.where(ytr == 1)[0]; idx0 = np.where(ytr == 0)[0]
    if len(idx1) < len(idx0):
        idx1 = rng.choice(idx1, len(idx0), replace=True)
    elif len(idx0) < len(idx1):
        idx0 = rng.choice(idx0, len(idx1), replace=True)
    idx = np.concatenate([idx0, idx1]); rng.shuffle(idx)
    sc = StandardScaler().fit(Xtr[idx])
    clf = MLPClassifier(hidden_layer_sizes=hidden, activation="relu", alpha=1e-3, learning_rate_init=1e-3,
                        max_iter=400, random_state=seed)
    clf.fit(sc.transform(Xtr[idx]), ytr[idx])
    return [clf.predict(sc.transform(Xte)) for Xte in Xte_list]

def metrics(y, yp):
    from sklearn.metrics import accuracy_score, f1_score
    return {"acc": float(accuracy_score(y, yp)), "f1": float(f1_score(y, yp, average="macro")),
            "f1_stress": float(f1_score(y, yp, pos_label=1, zero_division=0))}

def paired_stats(a, b):
    """a, b: per-subject arrays (a = config, b = reference). Returns dict of paired statistics."""
    d = np.asarray(a) - np.asarray(b)
    out = {"mean_diff": float(d.mean()), "std_diff": float(d.std(ddof=1)) if len(d) > 1 else 0.0}
    if len(d) > 1 and np.any(d != 0):
        out["p_wilcoxon"] = float(stats.wilcoxon(a, b).pvalue)
        out["p_ttest"] = float(stats.ttest_rel(a, b).pvalue)
    else:
        out["p_wilcoxon"] = 1.0; out["p_ttest"] = 1.0
    return out

def ci95(v):
    v = np.asarray(v, float)
    if len(v) < 2:
        return 0.0
    return float(stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / math.sqrt(len(v)))

# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wesad_dir", required=True)
    ap.add_argument("--out", default="results")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--quick", action="store_true", help="2 subjects, 1 seed, 2 eps (smoke test)")
    ap.add_argument("--n_jobs", type=int, default=-1)
    ap.add_argument("--rewrite_tex", action="store_true", help="only regenerate .tex outputs from <out>/results.json")
    args = ap.parse_args()
    if args.rewrite_tex:
        with open(os.path.join(args.out, "results.json")) as f:
            R = json.load(f)
        write_tex(R, args.out); print("tex outputs regenerated from results.json"); return
    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    subjects = SUBJECTS_ALL[:3] if args.quick else SUBJECTS_ALL
    eps_grid = [0.01, 0.1] if args.quick else EPS_GRID
    seeds = 1 if args.quick else args.seeds
    log = open(os.path.join(args.out, "run.log"), "w")
    def P(*a):
        s = " ".join(str(x) for x in a); print(s, flush=True); log.write(s + "\n"); log.flush()
    P(f"numba={HAVE_NUMBA} zfp={HAVE_ZFP} subjects={subjects} eps={eps_grid} seeds={seeds}")

    # ---- pass 1: load, compute global per-channel-group sigma (device scale constants)
    data = {}
    acc = {g: [0.0, 0.0, 0] for g in FS}
    for sid in subjects:
        ch, label, dur = load_subject(args.wesad_dir, sid)
        data[sid] = (ch, label, dur)
        for c in CHANNELS:
            g = CH_GROUP[c]; v = ch[c]
            acc[g][0] += v.sum(); acc[g][1] += (v ** 2).sum(); acc[g][2] += len(v)
        P(f"loaded S{sid}: {dur} s, labels {dict(Counter(label.tolist()))}")
    subject_ranges = {f"S{sid}": {c: [float(data[sid][0][c].min()), float(data[sid][0][c].max())] for c in CHANNELS} for sid in subjects}
    sigma = {g: math.sqrt(max(acc[g][1] / acc[g][2] - (acc[g][0] / acc[g][2]) ** 2, 0.0)) for g in FS}
    P("global sigma:", sigma)

    # ---- configs
    configs = [("raw", None), ("proposed", "native")]
    for e in eps_grid:
        configs += [("proposed", e), ("uniform", e)]
        if HAVE_ZFP:
            configs.append(("zfp", e))
    for D in DS_GRID:
        configs.append(("downsample", D))
    configs.append(("lossless", None))
    def cname(m, p):
        return m if p is None else (f"{m}_{p}" if isinstance(p, str) else f"{m}_{p:g}")

    # ---- pass 2: compression + verification + features
    comp = {}        # cname -> {"bits":..., "raw_bits":..., per-channel ...}
    feats = {}       # cname -> sid -> (X, y)
    verif = []       # rows for verification.txt
    openloop = []
    fnames = None
    for m, p in configs:
        name = cname(m, p)
        comp[name] = {"method": m, "param": p, "bits": 0.0, "raw_bits": 0.0, "max_err_ratio": 0.0,
                      "per_channel": {c: {"bits": 0.0, "raw_bits": 0.0, "max_err": 0.0, "eps_abs": None, "max_err_ratio": 0.0} for c in CHANNELS},
                      "n_esc": 0, "bound_ok": True}
        feats[name] = {}
        t0 = time.time()
        for sid in subjects:
            ch, label, dur = data[sid]
            rec = {}
            for c in CHANNELS:
                x = ch[c]; fs = FS[CH_GROUP[c]]
                if m == "raw":
                    xh, bits, err = x.copy(), RAW_BITS * len(x), 0.0
                    eps_abs = None
                else:
                    if m in ("proposed", "uniform", "zfp"):
                        eps_abs = NATIVE_EPS[CH_GROUP[c]] if p == "native" else p * sigma[CH_GROUP[c]]
                    else:
                        eps_abs = None
                    xh, bits, err, ex = compress_stream(x, m, eps_abs if eps_abs is not None else p, fs)
                    if m == "proposed":
                        comp[name]["n_esc"] += ex["n_esc"]
                rec[c] = xh
                pc = comp[name]["per_channel"][c]
                pc["bits"] += bits; pc["raw_bits"] += RAW_BITS * len(x); pc["max_err"] = max(pc["max_err"], err)
                pc["eps_abs"] = eps_abs
                if eps_abs is not None:
                    ratio = err / eps_abs
                    pc["max_err_ratio"] = max(pc["max_err_ratio"], ratio)
                    comp[name]["max_err_ratio"] = max(comp[name]["max_err_ratio"], ratio)
                    verif.append((name, sid, c, eps_abs, err, ratio))
                    if err > eps_abs * (1 + 1e-9):
                        comp[name]["bound_ok"] = False
                comp[name]["bits"] += bits; comp[name]["raw_bits"] += RAW_BITS * len(x)
            X, y, lab, fnames = window_features(rec, label, dur)
            feats[name][sid] = (X, y)
        comp[name]["ratio"] = comp[name]["raw_bits"] / comp[name]["bits"]
        comp[name]["bits_per_sample"] = RAW_BITS / comp[name]["ratio"]
        for c in CHANNELS:
            pc = comp[name]["per_channel"][c]; pc["ratio"] = pc["raw_bits"] / pc["bits"]
        P(f"[{name}] ratio={comp[name]['ratio']:.2f}x  bits/sample={comp[name]['bits_per_sample']:.2f}  "
          f"max_err/eps={comp[name]['max_err_ratio']:.4f} bound_ok={comp[name]['bound_ok']}  ({time.time()-t0:.0f}s)")
    # open-loop (buggy) predictor diagnostic
    for e in eps_grid:
        worst = 0.0
        for sid in subjects:
            ch, _, _ = data[sid]
            for c in CHANNELS:
                eps_abs = e * sigma[CH_GROUP[c]]
                _, _, err, _ = compress_stream(ch[c], "openloop", eps_abs, FS[CH_GROUP[c]])
                worst = max(worst, err / eps_abs)
        openloop.append({"eps_rel": e, "max_err_over_eps": worst})
        P(f"[openloop eps={e}] max_err/eps = {worst:.1f}  (bound violated: {worst > 1})")

    with open(os.path.join(args.out, "verification.txt"), "w") as f:
        f.write("config\tsubject\tchannel\teps_abs\tmax_abs_err\tmax_err/eps\n")
        for r in verif:
            f.write(f"{r[0]}\tS{r[1]}\t{r[2]}\t{r[3]:.6g}\t{r[4]:.6g}\t{r[5]:.6f}\n")
        f.write("\nSUMMARY (max over subjects and channels of max_abs_err/eps; must be <= 1):\n")
        for name in comp:
            if comp[name]["param"] is not None and comp[name]["method"] in ("proposed", "uniform", "zfp"):
                f.write(f"{name}\t{comp[name]['max_err_ratio']:.6f}\t{'OK' if comp[name]['bound_ok'] else 'VIOLATED'}\n")
        f.write("\nOpen-loop (original-sample) predictor diagnostic:\n")
        for r in openloop:
            f.write(f"eps_rel={r['eps_rel']}\tmax_err/eps={r['max_err_over_eps']:.2f}\n")
    violated = [n for n in comp if comp[n]["method"] in ("proposed", "uniform") and not comp[n]["bound_ok"]]
    if violated:
        P(f"ERROR: error bound violated for {violated}; see verification.txt. No LaTeX outputs written.")
        raise SystemExit(2)
    n_win = sum(len(feats["raw"][s][1]) for s in subjects)
    n_stress = sum(int(feats["raw"][s][1].sum()) for s in subjects)
    P(f"windows: {n_win} total, {n_stress} stress ({100*n_stress/n_win:.1f}%), {len(fnames)} features")

    # ---- pass 3: LOSO classification
    from joblib import Parallel, delayed
    names = list(comp.keys())
    def one_job(name, sid, seed):
        Xtr = np.vstack([feats[name][s][0] for s in subjects if s != sid])
        ytr = np.concatenate([feats[name][s][1] for s in subjects if s != sid])
        Xte, yte = feats[name][sid]
        # matched: train on this config; mismatched: raw-trained model tested on all configs (done when name=='raw')
        if name == "raw":
            preds = fit_eval(Xtr, ytr, [feats[n][sid][0] for n in names], seed)
            return {"name": name, "sid": sid, "seed": seed,
                    "matched": metrics(yte, preds[names.index("raw")]),
                    "mismatched": {n: metrics(feats[n][sid][1], preds[i]) for i, n in enumerate(names)}}
        preds = fit_eval(Xtr, ytr, [Xte], seed)
        return {"name": name, "sid": sid, "seed": seed, "matched": metrics(yte, preds[0])}
    jobs = [(n, s, k) for n in names for s in subjects for k in range(seeds)]
    P(f"running {len(jobs)} LOSO fits ...")
    t0 = time.time()
    res = Parallel(n_jobs=args.n_jobs, verbose=0)(delayed(one_job)(*j) for j in jobs)
    P(f"LOSO done in {time.time()-t0:.0f}s")

    # ---- aggregate: per subject (mean over seeds) -> mean/CI across subjects; paired tests vs raw
    per_subj = {n: {s: {k: [] for k in ("acc", "f1", "f1_stress")} for s in subjects} for n in names}
    per_subj_mm = {n: {s: {k: [] for k in ("acc", "f1", "f1_stress")} for s in subjects} for n in names}
    for r in res:
        for k in per_subj[r["name"]][r["sid"]]:
            per_subj[r["name"]][r["sid"]][k].append(r["matched"][k])
        if r["name"] == "raw":
            for n in names:
                for k in per_subj_mm[n][r["sid"]]:
                    per_subj_mm[n][r["sid"]][k].append(r["mismatched"][n][k])
    def summarize(table):
        out = {}
        ref = {k: np.array([np.mean(table["raw"][s][k]) for s in subjects]) for k in ("acc", "f1", "f1_stress")}
        for n in names:
            out[n] = {}
            for k in ("acc", "f1", "f1_stress"):
                v = np.array([np.mean(table[n][s][k]) for s in subjects])
                out[n][k] = {"mean": float(v.mean()), "std": float(v.std(ddof=1)) if len(v) > 1 else 0.0,
                             "ci95": ci95(v), "per_subject": {f"S{s}": float(x) for s, x in zip(subjects, v)},
                             "vs_raw": paired_stats(v, ref[k])}
        return out
    cls_matched = summarize(per_subj)
    cls_mismatched = summarize(per_subj_mm)

    # ---- matched-ratio comparison proposed vs downsampling
    matched_pairs = []
    for D in DS_GRID:
        dn = cname("downsample", D)
        props = [n for n in names if comp[n]["method"] == "proposed" and comp[n]["param"] != "native"]
        above = [n for n in props if comp[n]["ratio"] >= comp[dn]["ratio"]]
        best = min(above, key=lambda n: comp[n]["ratio"]) if above else max(props, key=lambda n: comp[n]["ratio"])
        matched_pairs.append({"downsample": dn, "ds_ratio": comp[dn]["ratio"], "ds_acc": cls_matched[dn]["acc"]["mean"],
                              "proposed": best, "prop_ratio": comp[best]["ratio"], "prop_acc": cls_matched[best]["acc"]["mean"],
                              "paired": paired_stats([cls_matched[best]["acc"]["per_subject"][f"S{s}"] for s in subjects],
                                                     [cls_matched[dn]["acc"]["per_subject"][f"S{s}"] for s in subjects])})

    # ---- energy / byte-rate derived numbers
    samples_per_s = 64 + 4 + 4 + 3 * 32
    raw_bytes_per_hour = samples_per_s * RAW_BITS / 8 * 3600
    energy = {n: {"ratio": comp[n]["ratio"], "bytes_per_hour": raw_bytes_per_hour / comp[n]["ratio"],
                  "tx_energy_saving_pct": 100 * (1 - 1 / comp[n]["ratio"])} for n in names}

    results = {
        "meta": {"date": time.strftime("%Y-%m-%d %H:%M"), "python": sys.version.split()[0], "numpy": np.__version__,
                 "platform": platform.platform(), "numba": HAVE_NUMBA, "zfp": HAVE_ZFP, "subjects": subjects,
                 "n_subjects": len(subjects), "eps_grid": eps_grid, "ds_grid": DS_GRID, "seeds": seeds,
                 "win_sec": WIN_SEC, "step_sec": STEP_SEC, "tx_win_sec": TX_WIN_SEC, "raw_bits": RAW_BITS,
                 "n_windows": n_win, "n_stress_windows": n_stress, "n_features": len(fnames), "features": fnames,
                 "mlp_hidden": [64, 32], "runtime_s": time.time() - t_start, "quick": args.quick},
        "sigma": sigma, "subject_ranges": subject_ranges, "compression": comp, "openloop_diagnostic": openloop,
        "classification_matched": cls_matched, "classification_mismatched": cls_mismatched,
        "matched_ratio_pairs": matched_pairs, "energy": energy,
    }
    with open(os.path.join(args.out, "results.json"), "w") as f:
        json.dump(results, f, indent=1)
    write_tex(results, args.out)
    write_summary(results, args.out, P)
    P(f"total runtime {time.time()-t_start:.0f}s")

# ----------------------------------------------------------------------------- LaTeX outputs
def _fmt(x, nd=1):
    return f"{x:.{nd}f}"

def pfmt(p):
    return "$<$0.01" if p < 0.005 else f"{p:.2f}"

def write_tex(R, out):
    comp, clsM, clsU = R["compression"], R["classification_matched"], R["classification_mismatched"]
    M = R["meta"]
    def macro(name, val):
        return f"\\newcommand{{\\{name}}}{{{val}}}\n"
    def key(e):  # eps 0.01 -> "A", etc. for macro names
        return {0.005: "Ea", 0.01: "Eb", 0.02: "Ec", 0.05: "Ed", 0.1: "Ee", 0.2: "Ef"}.get(e, "Ex")
    lines = []
    lines.append(macro("nSubjects", M["n_subjects"]))
    lines.append(macro("nWindows", M["n_windows"]))
    lines.append(macro("nStressWindows", M["n_stress_windows"]))
    lines.append(macro("pctStress", _fmt(100 * M["n_stress_windows"] / M["n_windows"])))
    lines.append(macro("nFeatures", M["n_features"]))
    lines.append(macro("nSeeds", M["seeds"]))
    lines.append(macro("rawAcc", _fmt(100 * clsM["raw"]["acc"]["mean"])))
    lines.append(macro("rawAccCI", _fmt(100 * clsM["raw"]["acc"]["ci95"])))
    lines.append(macro("rawF", _fmt(100 * clsM["raw"]["f1"]["mean"])))
    for g, s in R["sigma"].items():
        lines.append(macro(f"sigma{g}", f"{s:.3g}"))
    lines.append(macro("ratioNative", _fmt(comp["proposed_native"]["ratio"])))
    lines.append(macro("bpsNative", _fmt(comp["proposed_native"]["bits_per_sample"], 2)))
    lines.append(macro("accNative", _fmt(100 * clsM["proposed_native"]["acc"]["mean"])))
    for n, c in comp.items():
        if c["method"] == "proposed" and c["param"] != "native":
            k = key(c["param"])
            lines.append(macro(f"ratio{k}", _fmt(c["ratio"])))
            lines.append(macro(f"bps{k}", _fmt(c["bits_per_sample"], 2)))
            lines.append(macro(f"acc{k}", _fmt(100 * clsM[n]["acc"]["mean"])))
            lines.append(macro(f"dAcc{k}", _fmt(100 * clsM[n]["acc"]["vs_raw"]["mean_diff"])))
            lines.append(macro(f"pAcc{k}", pfmt(clsM[n]['acc']['vs_raw']['p_wilcoxon'])))
            lines.append(macro(f"f{k}", _fmt(100 * clsM[n]["f1"]["mean"])))
            lines.append(macro(f"accMM{k}", _fmt(100 * clsU[n]["acc"]["mean"])))
            lines.append(macro(f"saving{k}", _fmt(R["energy"][n]["tx_energy_saving_pct"])))
            lines.append(macro(f"maxErrRatio{k}", f"{c['max_err_ratio']:.4f}"))
    lines.append(macro("ratioLossless", _fmt(comp["lossless"]["ratio"], 2)))
    lines.append(macro("savingLossless", _fmt(R["energy"]["lossless"]["tx_energy_saving_pct"])))
    props = [n for n in comp if comp[n]["method"] == "proposed" and comp[n]["param"] != "native"]
    lines.append(macro("maxErrRatioAll", f"{max(comp[n]['max_err_ratio'] for n in props):.6f}"))
    zf = [n for n in comp if comp[n]["method"] == "zfp"]
    if zf:
        lines.append(macro("zfpRatioMin", _fmt(min(comp[n]["ratio"] for n in zf))))
        lines.append(macro("zfpRatioMax", _fmt(max(comp[n]["ratio"] for n in zf))))
        lines.append(macro("zfpErrMin", _fmt(min(comp[n]["max_err_ratio"] for n in zf), 2)))
        lines.append(macro("zfpErrMax", _fmt(max(comp[n]["max_err_ratio"] for n in zf), 2)))
        lines.append(macro("zfpDAccMin", f"{min(100*clsM[n]['acc']['vs_raw']['mean_diff'] for n in zf):+.1f}"))
        lines.append(macro("zfpDAccMax", f"{max(100*clsM[n]['acc']['vs_raw']['mean_diff'] for n in zf):+.1f}"))
    gains = [comp[n]["ratio"] / comp["uniform_" + n.split("_")[1]]["ratio"] for n in props if ("uniform_" + n.split("_")[1]) in comp]
    if gains:
        lines.append(macro("predGainMin", _fmt(min(gains))))
        lines.append(macro("predGainMax", _fmt(max(gains))))
    eda_ranges = [v["EDA"][1] - v["EDA"][0] for v in R["subject_ranges"].values()]
    lines.append(macro("minEDARange", _fmt(min(eda_ranges), 2)))
    lines.append(macro("maxEDARange", _fmt(max(eda_ranges), 1)))
    for n in props:
        lines.append(macro(f"epsEDA{key(comp[n]['param'])}", _fmt(comp[n]["per_channel"]["EDA"]["eps_abs"], 2)))
        lines.append(macro(f"epsBVP{key(comp[n]['param'])}", _fmt(comp[n]["per_channel"]["BVP"]["eps_abs"], 2)))
        lines.append(macro(f"epsTEMP{key(comp[n]['param'])}", _fmt(comp[n]["per_channel"]["TEMP"]["eps_abs"], 3)))
        lines.append(macro(f"epsACC{key(comp[n]['param'])}", _fmt(comp[n]["per_channel"]["ACC_x"]["eps_abs"], 2)))
    lines.append(macro("pctSamplesBVPACC", _fmt(100 * (64 + 96) / 168)))
    def sci(v):
        e = int(math.floor(math.log10(v))); m = v / 10 ** e
        return f"{m:.1f}\\times10^{{{e}}}"
    lines.append(macro("maxOpenLoopSci", sci(max(r["max_err_over_eps"] for r in R["openloop_diagnostic"]))))
    lines.append(macro("minOpenLoopSci", sci(min(r["max_err_over_eps"] for r in R["openloop_diagnostic"]))))
    lines.append(macro("nSamplesSci", sci(comp["raw"]["raw_bits"] / RAW_BITS)))
    lines.append(macro("maxOpenLoop", _fmt(max(r["max_err_over_eps"] for r in R["openloop_diagnostic"]), 0)))
    lines.append(macro("minOpenLoop", _fmt(min(r["max_err_over_eps"] for r in R["openloop_diagnostic"]), 0)))
    for pr in R["matched_ratio_pairs"]:
        D = pr["downsample"].split("_")[1]
        kk = 'ABCDEFGH'[int(D) // 2]
        lines.append(macro(f"dsAcc{kk}", _fmt(100 * pr["ds_acc"])))
        lines.append(macro(f"dsPairEps{kk}", f"{comp[pr['proposed']]['param']:g}"))
        lines.append(macro(f"dsPairRatio{kk}", _fmt(pr["prop_ratio"])))
        lines.append(macro(f"dsPairAcc{kk}", _fmt(100 * pr["prop_acc"])))
        lines.append(macro(f"dsPairDiff{kk}", f"{100 * pr['paired']['mean_diff']:+.1f}"))
        lines.append(macro(f"dsPairP{kk}", f"{pr['paired']['p_wilcoxon']:.2f}"))
        lines.append(macro(f"dsAccMM{kk}", _fmt(100 * clsU[pr["downsample"]]["acc"]["mean"])))
        lines.append(macro(f"dsPairAbsDiff{kk}", _fmt(abs(100 * pr["paired"]["mean_diff"]))))
        lines.append(macro(f"dsP{kk}", f"{clsM[pr['downsample']]['acc']['vs_raw']['p_wilcoxon']:.2f}"))
        lines.append(macro(f"dsDAcc{kk}", f"{100 * clsM[pr['downsample']]['acc']['vs_raw']['mean_diff']:+.1f}"))
    with open(os.path.join(out, "macros.tex"), "w") as f:
        f.writelines(lines)

    # main table: proposed at each eps (ratio, bits/sample, acc, dAcc, p, F1, mismatched acc)
    rows = []
    rows.append("\\begin{tabular}{lccccccc}\n\\toprule\n")
    rows.append("Config. & $\\epsilon/\\sigma$ & Ratio $\\rho$ & bits/samp. & Acc. (\\%) & $\\Delta$Acc. & $p$ & F1 (\\%) \\\\\n\\midrule\n")
    c = clsM["raw"]
    rows.append(f"Uncompressed & --- & 1.0$\\times$ & 32.00 & {_fmt(100*c['acc']['mean'])} $\\pm$ {_fmt(100*c['acc']['ci95'])} & --- & --- & {_fmt(100*c['f1']['mean'])} \\\\\n")
    for n, cc in comp.items():
        if cc["method"] == "proposed" and cc["param"] != "native":
            m = clsM[n]
            rows.append(f"Proposed & {cc['param']:g} & {_fmt(cc['ratio'])}$\\times$ & {_fmt(cc['bits_per_sample'],2)} & "
                        f"{_fmt(100*m['acc']['mean'])} $\\pm$ {_fmt(100*m['acc']['ci95'])} & {100*m['acc']['vs_raw']['mean_diff']:+.1f} & "
                        f"{pfmt(m['acc']['vs_raw']['p_wilcoxon'])} & {_fmt(100*m['f1']['mean'])} \\\\\n")
    rows.append("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(out, "table_main.tex"), "w") as f:
        f.writelines(rows)

    # baseline table: compact eps-by-method grid + non-bounded methods
    props = [n for n in comp if comp[n]["method"] == "proposed" and comp[n]["param"] != "native"]
    eps_list = [comp[n]["param"] for n in props]
    ncol = len(eps_list)
    rows = ["\\begin{tabular}{l" + "c" * ncol + "}\n\\toprule\n",
            "$\\epsilon/\\sigma$ & " + " & ".join(f"{e:g}" for e in eps_list) + " \\\\\n\\midrule\n"]
    def row(lbl, vals):
        rows.append(lbl + " & " + " & ".join(vals) + " \\\\\n")
    def get(m, e):
        n = f"{m}_{e:g}"; return comp.get(n), clsM.get(n)
    row("Proposed: ratio $\\rho$", [f"{_fmt(get('proposed',e)[0]['ratio'])}$\\times$" for e in eps_list])
    row("Proposed: $\\Delta$Acc.", [f"{100*get('proposed',e)[1]['acc']['vs_raw']['mean_diff']:+.1f}" for e in eps_list])
    row("Scalar quant.: ratio $\\rho$", [f"{_fmt(get('uniform',e)[0]['ratio'])}$\\times$" for e in eps_list])
    row("Scalar quant.: $\\Delta$Acc.", [f"{100*get('uniform',e)[1]['acc']['vs_raw']['mean_diff']:+.1f}" for e in eps_list])
    if HAVE_ZFP and get('zfp', eps_list[0])[0] is not None:
        row("ZFP: ratio $\\rho$", [f"{_fmt(get('zfp',e)[0]['ratio'])}$\\times$" for e in eps_list])
        row("ZFP: realized $\\max|\\hat{x}-x|/\\epsilon$", [f"{get('zfp',e)[0]['max_err_ratio']:.2f}" for e in eps_list])
        row("ZFP: $\\Delta$Acc.", [f"{100*get('zfp',e)[1]['acc']['vs_raw']['mean_diff']:+.1f}" for e in eps_list])
    rows.append("\\midrule\n")
    rows.append("\\multicolumn{" + str(ncol + 1) + "}{l}{\\emph{Not error-bounded / reference:} ratio, $\\Delta$Acc. matched, Acc. mismatched} \\\\\n")
    for D in R["meta"]["ds_grid"]:
        n = f"downsample_{D}"
        rows.append(f"Downsampling $D={D}$ & \\multicolumn{{{ncol}}}{{l}}{{{_fmt(comp[n]['ratio'])}$\\times$, {100*clsM[n]['acc']['vs_raw']['mean_diff']:+.1f}, {_fmt(100*clsU[n]['acc']['mean'])}\\%}} \\\\\n")
    rows.append(f"Lossless zlib & \\multicolumn{{{ncol}}}{{l}}{{{_fmt(comp['lossless']['ratio'])}$\\times$, {100*clsM['lossless']['acc']['vs_raw']['mean_diff']:+.1f}, {_fmt(100*clsU['lossless']['acc']['mean'])}\\%}} \\\\\n")
    rows.append(f"Proposed, native resolution & \\multicolumn{{{ncol}}}{{l}}{{{_fmt(comp['proposed_native']['ratio'])}$\\times$, {100*clsM['proposed_native']['acc']['vs_raw']['mean_diff']:+.1f}, {_fmt(100*clsU['proposed_native']['acc']['mean'])}\\%}} \\\\\n")
    rows.append("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(out, "table_baselines.tex"), "w") as f:
        f.writelines(rows)

    # per-channel table for proposed at each eps
    props = [n for n in comp if comp[n]["method"] == "proposed" and comp[n]["param"] != "native"]
    rows = ["\\begin{tabular}{l" + "c" * (len(props) + 1) + "}\n\\toprule\n",
            "Channel & $\\sigma$ (unit) & " + " & ".join(f"$\\epsilon={comp[n]['param']:g}\\sigma$" for n in props) + " \\\\\n\\midrule\n"]
    units = {"BVP": "a.u.", "EDA": "$\\mu$S", "TEMP": "$^\\circ$C", "ACC": "1/64\\,g"}
    for ch in CHANNELS:
        g = CH_GROUP[ch]
        rows.append(f"{ch.replace('_', ' ')} & {R['sigma'][g]:.3g} {units[g]} & " + " & ".join(f"{_fmt(comp[n]['per_channel'][ch]['ratio'])}$\\times$" for n in props) + " \\\\\n")
    rows.append("\\midrule\nAll (rate-weighted) & & " + " & ".join(f"{_fmt(comp[n]['ratio'])}$\\times$" for n in props) + " \\\\\n")
    rows.append("Max $|\\hat{x}-x|/\\epsilon$ & & " + " & ".join(f"{comp[n]['max_err_ratio']:.3f}" for n in props) + " \\\\\n")
    rows.append("\\bottomrule\n\\end{tabular}\n")
    with open(os.path.join(out, "table_channels.tex"), "w") as f:
        f.writelines(rows)

def write_summary(R, out, P):
    comp, clsM, clsU = R["compression"], R["classification_matched"], R["classification_mismatched"]
    L = []
    L.append(f"WESAD compression experiments  ({R['meta']['date']}, {R['meta']['n_subjects']} subjects, {R['meta']['seeds']} seeds, {R['meta']['n_windows']} windows, runtime {R['meta']['runtime_s']/60:.1f} min)")
    L.append(f"sigma per channel group: " + ", ".join(f"{g}={s:.4g}" for g, s in R["sigma"].items()))
    L.append(f"uncompressed LOSO accuracy {100*clsM['raw']['acc']['mean']:.1f} +- {100*clsM['raw']['acc']['ci95']:.1f} (95% CI), macro-F1 {100*clsM['raw']['f1']['mean']:.1f}")
    L.append("")
    L.append(f"{'config':<18}{'ratio':>7}{'b/samp':>8}{'maxerr/eps':>11}{'acc':>7}{'dAcc':>7}{'p_wil':>7}{'F1':>7}{'acc_mm':>8}")
    for n, c in comp.items():
        m = clsM[n]; u = clsU[n]
        L.append(f"{n:<18}{c['ratio']:>7.2f}{c['bits_per_sample']:>8.2f}{c['max_err_ratio']:>11.4f}{100*m['acc']['mean']:>7.1f}"
                 f"{100*m['acc']['vs_raw']['mean_diff']:>+7.1f}{m['acc']['vs_raw']['p_wilcoxon']:>7.2f}{100*m['f1']['mean']:>7.1f}{100*u['acc']['mean']:>8.1f}")
    L.append("")
    L.append("Bound check (must be OK for proposed/uniform/zfp): " + ", ".join(f"{n}:{'OK' if c['bound_ok'] else 'VIOLATED'}" for n, c in comp.items() if c['method'] in ('proposed','uniform','zfp')))
    L.append("Open-loop predictor diagnostic (max_err/eps): " + ", ".join(f"eps={r['eps_rel']}:{r['max_err_over_eps']:.1f}" for r in R["openloop_diagnostic"]))
    L.append("Matched-ratio pairs (proposed vs downsampling):")
    for pr in R["matched_ratio_pairs"]:
        L.append(f"  {pr['downsample']} ({pr['ds_ratio']:.1f}x, acc {100*pr['ds_acc']:.1f}) vs {pr['proposed']} ({pr['prop_ratio']:.1f}x, acc {100*pr['prop_acc']:.1f})  diff {100*pr['paired']['mean_diff']:+.1f}  p_wilcoxon={pr['paired']['p_wilcoxon']:.3f}")
    txt = "\n".join(L)
    with open(os.path.join(out, "summary.txt"), "w") as f:
        f.write(txt + "\n")
    P("\n" + txt)

if __name__ == "__main__":
    main()
