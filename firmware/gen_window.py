import sys, os, json, numpy as np
sys.path.insert(0, "..")
import wesad_compression_experiments as W
R = json.load(open("../results/results.json")); sigma = R["sigma"]
ch, label, dur = W.load_subject("../data/WESAD", 2); t0 = 1200
lines = ["/* generated: S2, t0=%d s, eps = 0.05*sigma */\n#include <stdint.h>\n" % t0]; names = []
for c in W.CHANNELS:
    fs = W.FS[W.CH_GROUP[c]]; seg = ch[c][t0*fs:(t0+60)*fs].astype(np.float32)
    nm = c.lower(); names.append((nm, len(seg), 0.05*sigma[W.CH_GROUP[c]]))
    lines.append(f"static const float win_{nm}[{len(seg)}] = {{" + ",".join(repr(float(v)) + "f" for v in seg) + "};\n")
lines.append("typedef struct { const float *x; int n; float eps; const char *name; } chan_t;\n")
lines.append("static const chan_t chans[] = {" + ",".join(f'{{win_{nm},{n},{repr(float(eps))}f,"{nm}"}}' for nm, n, eps in names) + "};\n#define NCHANS 6\n")
open("window_data.h", "w").writelines(lines)
