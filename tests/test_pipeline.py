import json

import numpy as np
import trimesh

from xrpipe.core import clean, decimate, guess_unit, keep_for_budget, surface_error
from xrpipe.pipeline import KeepPredictor, process_asset


def sphere():
    return trimesh.creation.icosphere(subdivisions=4)            # 5,120 faces


def test_clean_removes_duplicates_and_degenerates():
    m = sphere()
    f = np.vstack([m.faces, m.faces[:10], [[0, 0, 1]]])          # 10 duplicate faces + 1 degenerate
    dirty = trimesh.Trimesh(m.vertices, f, process=False)
    c, rep = clean(dirty)
    assert rep["degenerate_or_duplicate_faces_removed"] == 11 and c.is_watertight


def test_error_is_zero_for_identical_and_grows_with_decimation():
    m = sphere()
    assert surface_error(m, m.copy())[0] < 1e-9                  # the old point-to-point version gave ~0.8% here
    e_mild, e_hard = surface_error(m, decimate(m, 0.5))[0], surface_error(m, decimate(m, 0.02))[0]
    assert e_mild < e_hard


def test_binary_search_respects_budget():
    m = sphere()
    k = keep_for_budget(m, 0.002)
    assert surface_error(m, decimate(m, k))[0] <= 0.002 + 1e-6


def test_unit_guess():
    assert guess_unit(743.4, (0.3, 3.0)) == ("mm", True)          # engine CAD export in millimetres
    assert guess_unit(4.4, (3.0, 6.0)) == ("m", True)             # a car already in metres
    assert guess_unit(29.8, (0.05, 1.0))[1] is False             # ambiguous: flag for a human


def test_end_to_end(tmp_path):
    m = sphere(); m.apply_scale(500)                              # a 1 m object exported in mm
    src = tmp_path / "ball.glb"; trimesh.Scene([m]).export(src)
    pred = KeepPredictor(path=tmp_path / "missing.joblib")         # no model: uses binary search
    r = process_asset(src, tmp_path / "out", expected_size=(0.3, 3.0), predictor=pred)
    assert r["unit_detected"] == "mm" and abs(max(r["size_m"]) - 1.0) < 0.05
    assert r["triangles"]["LOD2"] < r["triangles"]["LOD1"] < r["triangles"]["LOD0"]
    detail = json.loads((tmp_path / "out" / "ball_report.json").read_text())["parts_detail"][0]
    assert detail["err_LOD1"] <= 0.002 and detail["err_LOD2"] <= 0.01
    for lod in ("LOD0", "LOD1", "LOD2"):
        assert (tmp_path / "out" / f"ball_{lod}.glb").exists()
