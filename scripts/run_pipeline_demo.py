"""Run the full pipeline on 6 assets (automotive and CAD first) and time model vs binary search on one.
python scripts/run_pipeline_demo.py -> out/*.glb, out/*_report.json, results/pipeline_demo.json"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xrpipe.pipeline import KeepPredictor, process_asset  # noqa: E402

ASSETS = {"CarConcept": (3, 6), "CesiumMilkTruck": (3, 10), "2CylinderEngine": (0.3, 3), "GearboxAssy": (0.1, 1.5),
          "ReciprocatingSaw": (0.2, 0.8), "ToyCar": (0.03, 0.3)}
pred = KeepPredictor()
rows = []
for a, size in ASSETS.items():
    r = process_asset(ROOT / "data" / f"{a}.glb", ROOT / "out", size, pred)
    rows.append({k: r[k] for k in ("asset", "parts", "seconds", "unit_detected", "unit_confident", "size_m", "triangles",
                                   "xr_budget_pass", "cleanup", "safety_fixups")})
    print(json.dumps(rows[-1]), flush=True)
t = time.time()
slow = process_asset(ROOT / "data" / "2CylinderEngine.glb", ROOT / "out" / "binary_search", (0.3, 3),
                     KeepPredictor(path=ROOT / "results" / "none.joblib"))
timing = {"asset": "2CylinderEngine", "model_seconds": next(r["seconds"] for r in rows if r["asset"] == "2CylinderEngine.glb"),
          "binary_search_seconds": slow["seconds"], "binary_search_triangles": slow["triangles"]}
(ROOT / "results" / "pipeline_demo.json").write_text(json.dumps({"assets": rows, "timing": timing}, indent=1))
print(json.dumps(timing))
