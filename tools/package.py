"""Assemble the submission folder ~/Desktop/Researchpaper: paper PDF + sources, arXiv zip, code, results, figures.
Usage: python tools/package.py"""
import os, shutil, zipfile, re, subprocess
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); DST = os.path.expanduser("~/Desktop/Researchpaper")
os.makedirs(DST, exist_ok=True)
# paper
pd = os.path.join(DST, "paper"); os.makedirs(pd, exist_ok=True)
for f in ("paper_journal.pdf", "paper_journal.tex"): shutil.copy(os.path.join(ROOT, f), pd)
shutil.copy(os.path.join(ROOT, "paper.pdf"), os.path.join(pd, "conference_version_4page.pdf"))
# results dirs used by the paper
res_dirs = ["results", "results_alloc", "results_dalia", "results_alloc_dalia", "results_cnn", "results_cnn_alloc", "results_nurse", "results_alloc_nurse", "results_fw"]
rd = os.path.join(DST, "results"); os.makedirs(rd, exist_ok=True)
for d in res_dirs:
    src = os.path.join(ROOT, d)
    if os.path.isdir(src):
        shutil.copytree(src, os.path.join(rd, d), dirs_exist_ok=True); shutil.copytree(src, os.path.join(pd, d), dirs_exist_ok=True)
# arXiv zip: tex + every input (macros/tables) + figures
tex = open(os.path.join(ROOT, "paper_journal.tex")).read()
inputs = re.findall(r"\\input\{([^}]+)\}", tex) + re.findall(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", tex)
ad = os.path.join(DST, "arxiv"); os.makedirs(ad, exist_ok=True)
with zipfile.ZipFile(os.path.join(ad, "arxiv_submission.zip"), "w", zipfile.ZIP_DEFLATED) as z:
    z.write(os.path.join(ROOT, "paper_journal.tex"), "paper_journal.tex")
    for p in inputs:
        z.write(os.path.join(ROOT, p), p)
# code
cd = os.path.join(DST, "code"); os.makedirs(cd, exist_ok=True)
for f in ("wesad_compression_experiments.py", "bound_allocation_experiments.py", "dalia_experiments.py", "nurse_experiments.py", "cnn_experiments.py", "cnn_allocation_experiments.py", "README.md", "LICENSE", "paper_journal.tex", "paper.tex"):
    shutil.copy(os.path.join(ROOT, f), cd)
os.makedirs(os.path.join(cd, "tools"), exist_ok=True)
for f in ("slim_dalia.py", "nurse_prepare.py", "package.py"): shutil.copy(os.path.join(ROOT, "tools", f), os.path.join(cd, "tools"))
fw = os.path.join(cd, "firmware"); os.makedirs(fw, exist_ok=True)
for f in os.listdir(os.path.join(ROOT, "firmware")):
    if f.endswith((".c", ".h", ".py", ".ld")) and f != "window_data.h": shutil.copy(os.path.join(ROOT, "firmware", f), fw)
for d in res_dirs:
    if os.path.isdir(os.path.join(ROOT, d)): shutil.copytree(os.path.join(ROOT, d), os.path.join(cd, d), dirs_exist_ok=True)
print("packaged to", DST)
for root, dirs, files in os.walk(DST):
    depth = root[len(DST):].count(os.sep)
    if depth <= 1: print("  " * depth + os.path.basename(root) + "/", len(files), "files")
