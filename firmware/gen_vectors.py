"""Export test vectors from the Python (float64) codec: S2 and S10, all 6 channels, all 60 s windows, eps in {0.01,0.05,0.2} sigma."""
import sys, os, struct, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import wesad_compression_experiments as W, json
R = json.load(open(os.path.join(os.path.dirname(__file__), "..", "results", "results.json")))
sigma = R["sigma"]; out = open(sys.argv[2], "wb"); nw = 0
for sid in [2, 10]:
    ch, label, dur = W.load_subject(sys.argv[1], sid)
    for c in W.CHANNELS:
        fs = W.FS[W.CH_GROUP[c]]; x = ch[c].astype(np.float32).astype(np.float64)   # device sees float32 samples
        for e in (0.01, 0.05, 0.2):
            eps = e * sigma[W.CH_GROUP[c]]
            for s in range(0, len(x) - 60 * fs + 1, 60 * fs):
                seg = x[s:s + 60 * fs]
                xh, bits, ex = W.encode_proposed_window(seg, eps)
                _, sym, esc = W._lorenzo_encode(seg, 2.0 * eps, W.KMAX)
                out.write(struct.pack("<i", len(seg))); out.write(struct.pack("<f", eps)); out.write(seg.astype(np.float32).tobytes())
                out.write(struct.pack("<I", int(bits))); out.write(sym.astype(np.int32).tobytes()); nw += 1
print("windows written:", nw)
