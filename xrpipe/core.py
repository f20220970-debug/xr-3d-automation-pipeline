"""Core mesh operations: load, clean, measure, decimate, compare.

Everything here is plain Python (trimesh + fast-simplification + NumPy/SciPy) so it runs headless in CI or
a render farm. The Blender and Unity integrations in integrations/ call the same ideas from inside those tools.
"""
from __future__ import annotations

import numpy as np
import trimesh

UNITS = {"m": 1.0, "cm": 0.01, "mm": 0.001, "in": 0.0254}


def load_parts(path, min_faces=1, bake_transforms=False):
    """Meshes in a glTF/GLB/OBJ file as {name: Trimesh}.
    bake_transforms=False: unique geometries in their own local coordinates (used to build the ML dataset,
    where every measure is relative to the part's own size).
    bake_transforms=True: every placed instance with its node transform applied (used for export, so parts
    keep their position and the file's real scale; a first version skipped this and mis-scaled ToyCar,
    whose node carries a 0.001 scale)."""
    scene = trimesh.load(path, force="scene")
    if not bake_transforms:
        return {k: g for k, g in scene.geometry.items() if isinstance(g, trimesh.Trimesh) and len(g.faces) >= min_faces}
    out = {}
    for node in scene.graph.nodes_geometry:
        T, gname = scene.graph[node]
        g = scene.geometry.get(gname)
        if isinstance(g, trimesh.Trimesh) and len(g.faces) >= min_faces:
            m = g.copy(); m.apply_transform(T)
            out[f"{node}"] = m
    return out


def clean(mesh: trimesh.Trimesh):
    """Standard cleanup a 3D artist would do by hand. Returns (clean mesh, report of what changed)."""
    m = mesh.copy()
    before = {"vertices": len(m.vertices), "faces": len(m.faces)}
    m.merge_vertices(merge_tex=False, merge_norm=True)
    nondeg = m.nondegenerate_faces(height=1e-12)
    m.update_faces(nondeg)
    m.update_faces(m.unique_faces())
    m.remove_unreferenced_vertices()
    try:
        trimesh.repair.fix_normals(m, multibody=True)
    except Exception:                                   # fix_normals can fail on very broken meshes; keep going
        pass
    edges = m.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    rep = {"vertices_before": before["vertices"], "vertices_after": len(m.vertices),
           "faces_before": before["faces"], "faces_after": len(m.faces),
           "degenerate_or_duplicate_faces_removed": before["faces"] - len(m.faces),
           "duplicate_vertices_merged": before["vertices"] - len(m.vertices),
           "non_manifold_edges": int((counts > 2).sum()), "boundary_edges": int((counts == 1).sum()),
           "watertight": bool(m.is_watertight)}
    return m, rep


def guess_unit(extent_max: float, expected_size_m: tuple[float, float]):
    """Pick the unit that puts the largest dimension inside the expected real-world size range.
    CAD exports are often in mm or cm while Unity/Unreal expect metres (Unity) or cm (Unreal).
    Returns (unit, confident). Not confident when several units fit, or none does: flag for a human."""
    lo, hi = expected_size_m
    fits = [u for u, s in UNITS.items() if lo <= extent_max * s <= hi]
    if len(fits) == 1:
        return fits[0], True
    centre = np.sqrt(lo * hi)
    best = min(UNITS, key=lambda u: abs(np.log(extent_max * UNITS[u] / centre)))
    return best, False


def decimate(mesh: trimesh.Trimesh, keep: float) -> trimesh.Trimesh:
    """Quadric edge-collapse decimation keeping roughly `keep` of the triangles."""
    import fast_simplification
    keep = float(np.clip(keep, 0.005, 1.0))
    if keep >= 0.999:
        return mesh.copy()
    v, f = fast_simplification.simplify(np.asarray(mesh.vertices, np.float32), np.asarray(mesh.faces, np.int64),
                                        target_reduction=1 - keep)
    return trimesh.Trimesh(v, f, process=False)


def surface_error(a: trimesh.Trimesh, b: trimesh.Trimesh, n=2500, seed=0):
    """Symmetric point-to-SURFACE deviation between two meshes, as a fraction of a's bounding-box diagonal.
    Samples points on each surface and measures the exact distance to the other surface (not to the other
    mesh's sample points: that version had a ~0.8% noise floor even for identical meshes; see tests).
    Returns (p95, max). p95 is robust to sampling noise and is what the LOD budget uses."""
    if len(b.faces) == 0:
        return 1.0, 1.0
    pa, _ = trimesh.sample.sample_surface(a, n, seed=seed)
    pb, _ = trimesh.sample.sample_surface(b, n, seed=seed + 1)
    _, da, _ = trimesh.proximity.closest_point(b, pa)
    _, db, _ = trimesh.proximity.closest_point(a, pb)
    d = np.concatenate([da, db])
    diag = float(np.linalg.norm(a.extents)) or 1.0
    return float(np.percentile(d, 95) / diag), float(d.max() / diag)


def keep_for_budget(mesh, budget, lo=0.005, hi=1.0, iters=7):
    """'Oracle' answer: smallest keep ratio whose p95 error stays within `budget`, by binary search on the
    real decimation (slow: ~8 decimations + error checks per part). This labels the ML training data."""
    if surface_error(mesh, decimate(mesh, lo))[0] <= budget:
        return lo
    for _ in range(iters):
        mid = np.sqrt(lo * hi)                         # search in log space: ratios span 0.5%..100%
        if surface_error(mesh, decimate(mesh, mid))[0] <= budget:
            hi = mid
        else:
            lo = mid
    return hi


def features(mesh: trimesh.Trimesh) -> dict:
    """Cheap shape descriptors (no decimation needed) used to predict how far a part can be simplified."""
    ext = np.sort(mesh.extents)[::-1] + 1e-12
    diag = float(np.linalg.norm(mesh.extents)) or 1.0
    ang = mesh.face_adjacency_angles if len(mesh.face_adjacency) else np.array([0.0])
    el = mesh.edges_unique_length if len(mesh.edges_unique) else np.array([0.0])
    edges = mesh.edges_sorted
    _, counts = np.unique(edges, axis=0, return_counts=True)
    bbox_area = 2 * (ext[0] * ext[1] + ext[1] * ext[2] + ext[0] * ext[2])
    return {"log_faces": float(np.log10(len(mesh.faces))), "log_verts": float(np.log10(max(len(mesh.vertices), 1))),
            "area_over_bbox_area": float(mesh.area / bbox_area),
            "mean_dihedral_deg": float(np.degrees(ang).mean()), "p90_dihedral_deg": float(np.degrees(np.percentile(ang, 90))),
            "frac_flat_edges": float((ang < np.radians(1)).mean()), "frac_sharp_edges": float((ang > np.radians(30)).mean()),
            "frac_boundary_edges": float((counts == 1).mean()), "mean_edge_over_diag": float(el.mean() / diag),
            "edge_len_cv": float(el.std() / (el.mean() + 1e-12)), "aspect_mid": float(ext[1] / ext[0]),
            "aspect_min": float(ext[2] / ext[0]), "faces_per_area": float(np.log10(len(mesh.faces) / (mesh.area / diag**2 + 1e-12))),
            "watertight": float(mesh.is_watertight)}
