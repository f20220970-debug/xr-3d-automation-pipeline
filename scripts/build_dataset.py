"""Label every part with the 'oracle' keep ratio for two error budgets, plus cheap shape features.
python scripts/build_dataset.py  -> results/parts_dataset.csv   (~15-25 min on 2 cores)"""
import glob
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from xrpipe.core import clean, features, keep_for_budget, load_parts  # noqa: E402

BUDGETS = {"lod1": 0.002, "lod2": 0.01}          # p95 deviation as a fraction of the part's diagonal
MAX_PARTS_PER_ASSET, MIN_F, MAX_F = 40, 200, 150_000


def job(args):
    asset, name, v, f = args
    import trimesh
    m, rep = clean(trimesh.Trimesh(v, f, process=False))
    if len(m.faces) < MIN_F:
        return None
    t = time.time()
    row = {"asset": asset, "part": name, "faces": len(m.faces), **{k: rep[k] for k in ("boundary_edges", "non_manifold_edges")},
           **features(m)}
    for k, b in BUDGETS.items():
        row[f"keep_{k}"] = keep_for_budget(m, b)
    row["oracle_seconds"] = time.time() - t
    return row


def main():
    rng = np.random.default_rng(0)
    jobs = []
    for path in sorted(glob.glob(str(ROOT / "data" / "*.glb"))):
        asset = Path(path).stem
        parts = [(k, g) for k, g in load_parts(path, MIN_F).items() if len(g.faces) <= MAX_F]
        if len(parts) > MAX_PARTS_PER_ASSET:
            parts = [parts[i] for i in rng.choice(len(parts), MAX_PARTS_PER_ASSET, replace=False)]
        jobs += [(asset, k, np.asarray(g.vertices), np.asarray(g.faces)) for k, g in parts]
    print(len(jobs), "parts", flush=True)
    rows = []
    with Pool(2) as pool:
        for i, r in enumerate(pool.imap_unordered(job, jobs, chunksize=2)):
            if r:
                rows.append(r)
            if i % 25 == 0:
                print(i, flush=True)
                pd.DataFrame(rows).to_csv(ROOT / "results" / "parts_dataset.csv", index=False)
    pd.DataFrame(rows).to_csv(ROOT / "results" / "parts_dataset.csv", index=False)
    print("done", len(rows))


if __name__ == "__main__":
    main()
