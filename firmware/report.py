"""Build the firmware evidence: (1) regenerate test vectors and the embedded window from the current WESAD results,
(2) native bit-exactness test of the single-precision C encoder vs the Python codec, (3) Cortex-M4 emulation
(Unicorn) instruction counts, (4) code/RAM size. Writes ../results_fw/{summary.txt, macros_fw.tex, report.json}.
Energy estimate uses the nRF52840 product specification figure of 52 uA/MHz (CoreMark from flash, 64 MHz) at 3.0 V,
stated as an estimate. Usage: python report.py
"""
import subprocess, re, json, os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
GCC = "/Users/Pratik/wesad_paper/tools/arm/arm-gnu-toolchain-13.3.rel1-darwin-arm64-arm-none-eabi/bin/arm-none-eabi-"
PY = "../.venv/bin/python"
out = "../results_fw"; os.makedirs(out, exist_ok=True)
def run(cmd): return subprocess.run(cmd, shell=True, capture_output=True, text=True).stdout
# 1. vectors + embedded window
run(f"{PY} gen_vectors.py ../data/WESAD vectors.bin")
run(f"{PY} gen_window.py")
# 2. native test
run("cc -O2 -o test_native codec.c test_native.c -lm")
nat = run("./test_native vectors.bin"); print(nat.strip())
m = re.search(r"windows=(\d+) samples=(\d+) identical_indices=([\d.]+) bits_c=(\d+) bits_py=(\d+) rel_diff=([-\d.]+) max_err_over_eps=([\d.]+) bound_violations=(\d+)", nat)
nwin, nsamp, ident, bc, bpy, rel, maxerr, viol = m.groups()
# 3. cross-compile + emulate
run(f"{GCC}gcc -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard -O2 -ffunction-sections -fdata-sections -nostartfiles -specs=nano.specs -specs=nosys.specs -T mps2.ld -Wl,--gc-sections -o fw_m4.elf codec.c main_emu.c -lm -lc -lgcc")
emu = run(f"MAXINSN=200000000 {PY} emulate_m4.py fw_m4.elf"); print(emu.strip())
tot = int(re.search(r"total=(\d+)", emu).group(1)); inside = int(re.search(r"inside_codec_encode=(\d+)", emu).group(1))
samples = int(re.search(r"TOTAL samples=(\d+)", emu).group(1)); bits = int(re.search(r"TOTAL samples=\d+ bits=(\d+)", emu).group(1))
# 4. sizes: codec text = codec.c functions (codec_encode, huff_lengths, codec_reconstruct, cmp) + libc pieces it needs (qsort, memset, floorf)
nm = run(f"{GCC}nm --size-sort -S fw_m4.elf")
sizes = {}
for line in nm.splitlines():
    p = line.split()
    if len(p) == 4 and p[2].lower() in ("t", "w"): sizes[p[3]] = int(p[1], 16)
codec_funcs = [k for k in sizes if k.startswith("codec_") or k in ("huff_lengths", "cmp_i32", "qsort", "memset", "floorf", "memcpy")]
code_bytes = sum(sizes[k] for k in codec_funcs)
ram_bytes = 5 * 3840 * 4 + 3840 * 1 + (3840 * 4 * 3 + 2 * 3840 * 4 + 3840 + 3840 * 4)   # ws + static scratch
# energy estimate
f_mhz = 64.0; ua_per_mhz = 52.0; volt = 3.0
t_window_s = inside / (f_mhz * 1e6)              # assumes 1 instruction per cycle (lower bound)
e_window_mj = ua_per_mhz * f_mhz * 1e-6 * volt * t_window_s * 1e3
rep = {"native": {"windows": int(nwin), "samples": int(nsamp), "identical_indices": float(ident), "bits_c": int(bc), "bits_py": int(bpy), "rel_diff": float(rel), "max_err_over_eps": float(maxerr), "bound_violations": int(viol)},
       "emulation": {"total_instructions": tot, "codec_instructions": inside, "samples": samples, "bits": bits, "instr_per_sample": inside / samples, "cpu": "Cortex-M4 (Unicorn 2.1.4 emulation, hard-float, -O2, GCC 13.3)"},
       "size": {"codec_code_bytes": code_bytes, "codec_funcs": {k: sizes[k] for k in codec_funcs}, "ram_scratch_bytes": ram_bytes},
       "energy_estimate": {"f_mhz": f_mhz, "ua_per_mhz": ua_per_mhz, "volt": volt, "cpi_assumed": 1.0, "t_window_ms": t_window_s * 1e3, "e_window_mj": e_window_mj, "duty_cycle_pct": 100 * t_window_s / 60}}
json.dump(rep, open(os.path.join(out, "report.json"), "w"), indent=1)
L = [f"\\newcommand{{\\fwIdentical}}{{{100*float(ident):.1f}}}\n", f"\\newcommand{{\\fwBitsRelDiff}}{{{100*abs(float(rel)):.2f}}}\n", f"\\newcommand{{\\fwMaxErr}}{{{float(maxerr):.7f}}}\n",
     f"\\newcommand{{\\fwTestWindows}}{{{nwin}}}\n", f"\\newcommand{{\\fwTestSamplesSci}}{{{float(nsamp)/1e6:.1f}\\times10^{{6}}}}\n",
     f"\\newcommand{{\\fwInstrPerSample}}{{{inside/samples:.0f}}}\n", f"\\newcommand{{\\fwInstrWindowSci}}{{{inside/1e6:.2f}\\times10^{{6}}}}\n", f"\\newcommand{{\\fwWindowSamples}}{{{samples}}}\n",
     f"\\newcommand{{\\fwCodeKB}}{{{code_bytes/1024:.1f}}}\n", f"\\newcommand{{\\fwRamKB}}{{{ram_bytes/1024:.0f}}}\n",
     f"\\newcommand{{\\fwTimeMs}}{{{t_window_s*1e3:.0f}}}\n", f"\\newcommand{{\\fwDuty}}{{{100*t_window_s/60:.2f}}}\n", f"\\newcommand{{\\fwEnergyMJ}}{{{e_window_mj:.2f}}}\n"]
open(os.path.join(out, "macros_fw.tex"), "w").writelines(L)
S = [f"Firmware evidence", f"native test: {nat.strip()}", f"emulation: codec instructions={inside} for {samples} samples ({inside/samples:.0f}/sample); total incl. startup+semihosting={tot}",
     f"code size (codec + qsort/memset/floorf): {code_bytes} B; scratch RAM: {ram_bytes} B", f"estimate @64 MHz, CPI=1, 52 uA/MHz, 3.0 V: {t_window_s*1e3:.1f} ms per 60 s window ({100*t_window_s/60:.2f}% duty), {e_window_mj:.2f} mJ per window"]
open(os.path.join(out, "summary.txt"), "w").write("\n".join(S) + "\n"); print("\n".join(S))
