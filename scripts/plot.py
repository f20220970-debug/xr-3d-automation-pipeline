"""python scripts/plot.py -> docs/results.png"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
d = pd.read_csv(ROOT / "results" / "parts_dataset.csv")
ev = pd.read_csv(ROOT / "results" / "keep_model_eval.csv")
fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))

ax[0].scatter(d.frac_boundary_edges, d.keep_lod1, s=12, alpha=.5, color="#1f77b4")
ax[0].set_xlabel("share of open (boundary) edges in the part"); ax[0].set_ylabel("triangles that must be kept (LOD1 budget)")
ax[0].set_yscale("log"); ax[0].set_title("1. How far each of %d real parts can be simplified" % len(d))

e = ev[(ev.lod == "lod1") & (ev.method.str.contains("shipped") | ev.method.str.startswith("oracle"))].copy()
e["label"] = e.method.str.replace(" + verify/step-up (shipped)", "", regex=False).map(
    {"model_median": "ML guess\n+ check", "model_q90": "ML (cautious)\n+ check", "fixed_median": "one fixed ratio\n+ check",
     "oracle (binary search)": "binary search\n(8 checks)"})
ax[1].bar(e.label, e.triangles_kept_pct_of_original, color=["#1f4e79", "#6baed6", "#e4572e", "#2e7d32"])
for i, (t, c) in enumerate(zip(e.triangles_kept_pct_of_original, e.avg_checks_per_part)):
    ax[1].text(i, t + 1, f"{t:.0f}%\n{c:.1f} checks/part", ha="center", fontsize=8)
ax[1].set_ylabel("% of original triangles shipped in LOD1"); ax[1].set_ylim(0, 65)
ax[1].set_title("2. All ship 100% within budget: cost vs triangles")

rep = [json.loads(p.read_text()) for p in sorted((ROOT / "out").glob("*_report.json"))]
if rep:
    names = [r["asset"].replace(".glb", "") for r in rep]
    for i, lod in enumerate(["LOD0", "LOD1", "LOD2"]):
        ax[2].bar([x + i * 0.27 for x in range(len(rep))], [r["triangles"][lod] / 1000 for r in rep], 0.27, label=lod)
    ax[2].set_xticks([x + 0.27 for x in range(len(rep))], names, rotation=20, fontsize=8)
    ax[2].set_ylabel("triangles (thousands)"); ax[2].legend(); ax[2].set_title("3. Pipeline output per asset")
plt.tight_layout(); plt.savefig(ROOT / "docs" / "results.png", dpi=130); print("saved")
