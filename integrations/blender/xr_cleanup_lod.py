"""Blender version of the pipeline's cleanup + LOD step (run headless, batch over a folder):

    blender -b -P integrations/blender/xr_cleanup_lod.py -- in_dir out_dir

For each GLB/FBX/OBJ: import, merge-by-distance, delete loose geometry, recalculate normals, apply transforms,
rescale to metres, then add LOD1/LOD2 copies with the Decimate modifier and export a GLB with *_LOD0/1/2
objects. The keep ratios can come from the Python model (results/keep_model.joblib) via a JSON sidecar.
NOTE: written against Blender 4.x's Python API; not executed in this repo's CI (no Blender there).
"""
import json
import sys
from pathlib import Path

import bpy

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
IN, OUT = Path(argv[0]), Path(argv[1])
OUT.mkdir(parents=True, exist_ok=True)
DEFAULT_KEEP = {"LOD1": 0.5, "LOD2": 0.15}


def reset():
    bpy.ops.wm.read_factory_settings(use_empty=True)


def import_any(p):
    ext = p.suffix.lower()
    if ext in (".glb", ".gltf"):
        bpy.ops.import_scene.gltf(filepath=str(p))
    elif ext == ".fbx":
        bpy.ops.import_scene.fbx(filepath=str(p))
    elif ext == ".obj":
        bpy.ops.wm.obj_import(filepath=str(p))


def clean(obj, scale):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.remove_doubles(threshold=1e-6)          # "merge by distance"
    bpy.ops.mesh.delete_loose()
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")
    obj.scale = [s * scale for s in obj.scale]
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True)


for p in sorted(IN.iterdir()):
    if p.suffix.lower() not in (".glb", ".gltf", ".fbx", ".obj"):
        continue
    reset(); import_any(p)
    side = p.with_suffix(".keep.json")
    keep = json.loads(side.read_text()) if side.exists() else {}
    scale = keep.get("scale", 1.0)
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    for o in meshes:
        bpy.ops.object.select_all(action="DESELECT")
        clean(o, scale)
        base = o.name
        o.name = f"{base}_LOD0"
        for lod in ("LOD1", "LOD2"):
            c = o.copy(); c.data = o.data.copy(); c.name = f"{base}_{lod}"
            bpy.context.collection.objects.link(c)
            m = c.modifiers.new("decimate", "DECIMATE")
            m.ratio = keep.get(base, {}).get(lod, DEFAULT_KEEP[lod])
    bpy.ops.export_scene.gltf(filepath=str(OUT / f"{p.stem}_xr.glb"), export_apply=True)
    print("exported", p.name)
