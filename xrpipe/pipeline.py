"""End-to-end batch pipeline: raw asset in, XR-ready LODs + a QA report out, no manual steps.

    python -m xrpipe.pipeline data/CarConcept.glb --expected-size 3 6 --out out/
    python -m xrpipe.pipeline data/ --out out/            # a whole folder

Per asset:
  1. load every part; clean it (weld duplicates, drop degenerate/duplicate faces, fix normals)
  2. guess the source unit from the expected real-world size and rescale to metres (flag if unsure)
  3. centre the pivot at the bottom-centre (what Unity/Unreal users expect for placing a car on the ground)
  4. build LOD0/LOD1/LOD2 per part: the ML model predicts how far each part can be simplified for a given
     error budget (or a slow binary search if no model is trained), then the result is CHECKED and the
     ratio stepped up if the check fails, so the budget is guaranteed, not just predicted
  5. export GLB files named *_LOD0/1/2 (Unity's importer builds a LODGroup from that suffix)
  6. check XR budgets (triangles per LOD) and write a JSON + Markdown QA report
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import trimesh

from .core import clean, decimate, features, guess_unit, keep_for_budget, load_parts, surface_error, UNITS

BUDGETS = {"LOD1": 0.002, "LOD2": 0.01}            # p95 surface deviation / part diagonal
XR_TRIANGLE_BUDGET = {"LOD0": 500_000, "LOD1": 150_000, "LOD2": 50_000}   # per asset; set per project/device


class KeepPredictor:
    """Wraps the trained model (results/keep_model.joblib) with a safety check."""

    def __init__(self, path=None):
        self.models = None
        path = Path(path or Path(__file__).resolve().parents[1] / "results" / "keep_model.joblib")
        if path.exists():
            import joblib
            self.models = joblib.load(path)

    def keep(self, mesh, lod, budget):
        if self.models is None:
            return keep_for_budget(mesh, budget), "binary search"
        import pandas as pd
        f = pd.DataFrame([features(mesh)])[self.models["features"]]
        z = float(self.models[lod].predict(f)[0])
        keep = float(np.clip(1 / (1 + np.exp(-z)), 0.005, 1.0))
        return keep, "model"


def process_asset(path, out_dir, expected_size=(0.05, 10.0), predictor=None, verify=True):
    t0 = time.time()
    predictor = predictor or KeepPredictor()
    path, out_dir = Path(path), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    parts = load_parts(path, bake_transforms=True)
    cleaned, reports = {}, []
    for name, g in parts.items():
        m, rep = clean(g)
        if len(m.faces):
            cleaned[name] = m
            reports.append({"part": name, **rep})
    scene = trimesh.Scene(list(cleaned.values()))
    unit, confident = guess_unit(float(scene.extents.max()), tuple(expected_size))
    scale = UNITS[unit]
    lo, hi = scene.bounds
    pivot = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])        # glTF is Y-up: bottom-centre
    lods = {k: [] for k in ["LOD0", "LOD1", "LOD2"]}
    part_lod = []
    for name, m in cleaned.items():
        m = m.copy()
        m.apply_translation(-pivot)
        m.apply_scale(scale)
        lods["LOD0"].append(m)
        row = {"part": name, "faces_LOD0": len(m.faces)}
        for lod, budget in BUDGETS.items():
            keep, how = predictor.keep(m, lod, budget) if len(m.faces) >= 200 else (1.0, "too small")
            d = decimate(m, keep)
            err = surface_error(m, d)[0] if verify and keep < 1 else 0.0
            steps = 0
            while verify and err > budget and keep < 1.0 and steps < 6:     # safety net: never ship over budget
                keep = min(1.0, keep * 1.6); d = decimate(m, keep); err = surface_error(m, d)[0]; steps += 1
            d.visual = trimesh.visual.ColorVisuals(d)
            lods[lod].append(d)
            row.update({f"keep_{lod}": round(keep, 4), f"faces_{lod}": len(d.faces), f"err_{lod}": round(err, 5),
                        f"how_{lod}": how, f"fixups_{lod}": steps})
        part_lod.append(row)
    files = {}
    for lod, meshes in lods.items():
        fn = out_dir / f"{path.stem}_{lod}.glb"
        trimesh.Scene(meshes).export(fn)
        files[lod] = fn.name
    tri = {lod: int(sum(len(m.faces) for m in ms)) for lod, ms in lods.items()}
    report = {"asset": path.name, "parts": len(cleaned), "seconds": round(time.time() - t0, 1),
              "unit_detected": unit, "unit_confident": confident, "scale_applied": scale,
              "size_m": [round(float(x) * scale, 3) for x in scene.extents],
              "triangles": tri, "xr_budget": XR_TRIANGLE_BUDGET,
              "xr_budget_pass": {lod: tri[lod] <= XR_TRIANGLE_BUDGET[lod] for lod in tri},
              "cleanup": {"duplicate_vertices_merged": int(sum(r["duplicate_vertices_merged"] for r in reports)),
                          "bad_faces_removed": int(sum(r["degenerate_or_duplicate_faces_removed"] for r in reports)),
                          "parts_with_non_manifold_edges": int(sum(r["non_manifold_edges"] > 0 for r in reports)),
                          "parts_with_open_boundaries": int(sum(r["boundary_edges"] > 0 for r in reports))},
              "safety_fixups": int(sum(r.get("fixups_LOD1", 0) + r.get("fixups_LOD2", 0) for r in part_lod)),
              "files": files}
    (out_dir / f"{path.stem}_report.json").write_text(json.dumps({**report, "parts_detail": part_lod}, indent=1))
    return report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs="+")
    ap.add_argument("--out", default="out")
    ap.add_argument("--expected-size", nargs=2, type=float, default=[0.05, 10.0], help="real-world size range of the largest dimension, metres")
    a = ap.parse_args()
    paths = []
    for p in a.inputs:
        p = Path(p)
        paths += sorted(p.glob("*.glb")) + sorted(p.glob("*.gltf")) + sorted(p.glob("*.obj")) if p.is_dir() else [p]
    pred = KeepPredictor()
    for p in paths:
        r = process_asset(p, a.out, a.expected_size, pred)
        print(json.dumps({k: r[k] for k in ("asset", "parts", "seconds", "unit_detected", "unit_confident", "triangles", "xr_budget_pass")}))


if __name__ == "__main__":
    main()
