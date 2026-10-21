"""Extract the nurse stress dataset (Hosseini et al. 2022) into per-nurse npz files of concatenated E4 sessions
with a 4 Hz label stream (-1 unlabeled, 0 no stress, 1 medium, 2 high) built from SurveyResults.xlsx.
Survey times are local; the UTC offset is chosen as the one (among US offsets) that maximizes the fraction of
survey intervals covered by recordings, and is written to meta.json.
Usage: python nurse_prepare.py Stress_dataset.zip SurveyResults.xlsx outdir
"""
import sys, io, zipfile, json, os, datetime as dt
import numpy as np, pandas as pd
zpath, xpath, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
FS = {"BVP": 64, "EDA": 4, "TEMP": 4, "ACC": 32}
GUARD = 60  # seconds of unlabeled guard at every session join

def read_csv(b, ncol=1):
    lines = b.decode().strip().splitlines()
    t0 = float(lines[0].split(",")[0]); fs = float(lines[1].split(",")[0])
    arr = np.array([[float(v) for v in l.split(",")] for l in lines[2:]], dtype=np.float32)
    return t0, fs, arr.reshape(-1, ncol)

z = zipfile.ZipFile(zpath)
sessions = {}
for n in sorted(z.namelist()):
    if not n.endswith(".zip"): continue
    nurse = n.split("/")[0]
    inner = zipfile.ZipFile(io.BytesIO(z.read(n)))
    try:
        t0b, fsb, bvp = read_csv(inner.read("BVP.csv")); t0e, fse, eda = read_csv(inner.read("EDA.csv"))
        t0t, fst, temp = read_csv(inner.read("TEMP.csv")); t0a, fsa, acc = read_csv(inner.read("ACC.csv"), 3)
    except Exception as e:
        print("skip", n, e); continue
    if min(len(bvp) / 64, len(eda) / 4, len(temp) / 4, len(acc) / 32) < 120: continue
    t0 = max(t0b, t0e, t0t, t0a); dur = int(min(t0b + len(bvp) / 64, t0e + len(eda) / 4, t0t + len(temp) / 4, t0a + len(acc) / 32) - t0)
    if dur < 120: continue
    sl = lambda a, ts, fs: a[int(round((t0 - ts) * fs)):int(round((t0 - ts) * fs)) + dur * fs]
    sessions.setdefault(nurse, []).append({"t0": t0, "dur": dur, "BVP": sl(bvp[:, 0], t0b, 64), "EDA": sl(eda[:, 0], t0e, 4), "TEMP": sl(temp[:, 0], t0t, 4), "ACC": sl(acc, t0a, 32)})
print("nurses:", {k: len(v) for k, v in sessions.items()})

x = pd.read_excel(xpath); x = x[x["Stress level"].isin([0, 1, 2])]
surv = []
for _, r in x.iterrows():
    d = pd.to_datetime(r["date"]).date(); st = r["Start time"]; en = r["End time"]
    s = dt.datetime.combine(d, st if isinstance(st, dt.time) else pd.to_datetime(str(st)).time()); e = dt.datetime.combine(d, en if isinstance(en, dt.time) else pd.to_datetime(str(en)).time())
    if e <= s: e += dt.timedelta(days=1)
    surv.append((str(r["ID"]), s, e, int(r["Stress level"])))
# choose UTC offset maximizing coverage of survey intervals by recordings
best = None
for off_h in (-4, -5, -6, -7):
    cov = 0
    for nid, s, e, lv in surv:
        su = (s - dt.timedelta(hours=off_h)).replace(tzinfo=dt.timezone.utc).timestamp(); eu = (e - dt.timedelta(hours=off_h)).replace(tzinfo=dt.timezone.utc).timestamp()
        for ses in sessions.get(nid, []):
            if ses["t0"] <= su and ses["t0"] + ses["dur"] >= eu: cov += 1; break
    print(f"offset UTC{off_h:+d}: {cov}/{len(surv)} survey intervals fully inside a recording")
    if best is None or cov > best[1]: best = (off_h, cov)
off_h = best[0]
# build per-nurse concatenated streams + 4 Hz label stream
meta = {"utc_offset_h": off_h, "coverage": best[1], "n_surveys": len(surv), "nurses": {}}
for nurse, ses_list in sessions.items():
    ses_list.sort(key=lambda s: s["t0"])
    cat = {k: [] for k in FS}; lab = []
    for ses in ses_list:
        n4 = ses["dur"] * 4; l = -np.ones(n4, dtype=np.int8)
        for nid, s, e, lv in surv:
            if nid != nurse: continue
            su = (s - dt.timedelta(hours=off_h)).replace(tzinfo=dt.timezone.utc).timestamp(); eu = (e - dt.timedelta(hours=off_h)).replace(tzinfo=dt.timezone.utc).timestamp()
            a = int(max(0, (su - ses["t0"]) * 4)); b = int(min(n4, (eu - ses["t0"]) * 4))
            if b > a: l[a:b] = lv
        l[:GUARD * 4] = -1; l[-GUARD * 4:] = -1
        for k in FS: cat[k].append(ses[k])
        lab.append(l)
    arrs = {k: np.concatenate(cat[k]) for k in FS}; L = np.concatenate(lab)
    np.savez_compressed(os.path.join(out, f"{nurse}.npz"), BVP=arrs["BVP"], EDA=arrs["EDA"], TEMP=arrs["TEMP"], ACC=arrs["ACC"], label=L)
    cnt = {int(v): int((L == v).sum()) / 4 for v in (0, 1, 2)}
    meta["nurses"][nurse] = {"sessions": len(ses_list), "hours": float(L.size / 4 / 3600), "labeled_seconds": cnt}
    print(nurse, meta["nurses"][nurse])
json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
