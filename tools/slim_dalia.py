import pickle, numpy as np, sys, os, subprocess
zipf, outdir = sys.argv[1], sys.argv[2]
for i in range(1, 16):
    name = f"PPG_FieldStudy/S{i}/S{i}.pkl"
    subprocess.run(["unzip", "-q", "-o", zipf, name, "-d", "."], check=True)
    with open(name, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    if i == 1:
        print("keys:", list(d.keys())); print("wrist:", {k: v.shape for k, v in d["signal"]["wrist"].items()})
        print("activity:", np.asarray(d["activity"]).shape, "label:", np.asarray(d["label"]).shape)
    w = d["signal"]["wrist"]
    np.savez_compressed(os.path.join(outdir, f"S{i}.npz"), BVP=w["BVP"][:, 0].astype(np.float32), EDA=w["EDA"][:, 0].astype(np.float32),
                        TEMP=w["TEMP"][:, 0].astype(np.float32), ACC=w["ACC"].astype(np.float32),
                        activity=np.asarray(d["activity"]).astype(np.int8).ravel(), hr=np.asarray(d["label"]).astype(np.float32))
    os.remove(name); print(f"S{i} slimmed", flush=True)
