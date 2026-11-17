"""Train and evaluate the 'how far can this part be simplified?' model.

Problem: hitting an error budget by binary search means ~7 decimations + error checks per part per LOD.
On an automotive asset with hundreds of parts that's the pipeline bottleneck. A model that predicts the keep
ratio from cheap shape features needs one decimation plus one check.

Evaluation (held-out ASSETS, not parts: GroupKFold by asset, so a car's parts never leak into its own test):
  - compliance: % of held-out parts whose predicted LOD actually meets the error budget
  - triangles kept: total triangles, vs the oracle (the minimum possible)
  - baselines: one fixed ratio for every part, chosen on the training folds (a) at the median, and
    (b) conservative enough to meet the budget on 90% of training parts
  - the model at a conservative quantile (90%) is compared with baseline (b) at matched compliance
Run: python scripts/train_keep_model.py -> results/keep_model_eval.csv, results/keep_model.joblib
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
FEATS = ["log_faces", "log_verts", "area_over_bbox_area", "mean_dihedral_deg", "p90_dihedral_deg", "frac_flat_edges",
         "frac_sharp_edges", "frac_boundary_edges", "mean_edge_over_diag", "edge_len_cv", "aspect_mid", "aspect_min",
         "faces_per_area", "watertight"]
logit = lambda p: np.log(p / (1 - p))
sig = lambda z: 1 / (1 + np.exp(-z))


def gbr(alpha=None):
    kw = dict(n_estimators=300, max_depth=3, learning_rate=0.05, subsample=0.8, random_state=0)
    return GradientBoostingRegressor(loss="quantile", alpha=alpha, **kw) if alpha else GradientBoostingRegressor(**kw)


def main():
    d = pd.read_csv(ROOT / "results" / "parts_dataset.csv").dropna(subset=FEATS)
    out, models = [], {"features": FEATS}
    for lod in ["lod1", "lod2"]:
        y = d[f"keep_{lod}"].clip(0.005, 0.999)
        z = logit(y)
        preds = {k: np.zeros(len(d)) for k in ["model_median", "model_q90", "fixed_median", "fixed_q90"]}
        for tr, te in GroupKFold(n_splits=5).split(d, z, d.asset):
            Xtr, Xte = d.iloc[tr][FEATS], d.iloc[te][FEATS]
            preds["model_median"][te] = sig(gbr(0.5).fit(Xtr, z.iloc[tr]).predict(Xte))
            preds["model_q90"][te] = sig(gbr(0.9).fit(Xtr, z.iloc[tr]).predict(Xte))
            preds["fixed_median"][te] = y.iloc[tr].median()
            preds["fixed_q90"][te] = y.iloc[tr].quantile(0.9)
        oracle_tris = float((d.faces * y).sum())
        for name, p in preds.items():
            # a prediction meets the budget if it keeps at least the oracle's minimum ratio (monotone in keep)
            ok = p >= y.values * 0.999
            out.append({"lod": lod, "method": name, "compliance_pct": round(100 * ok.mean(), 1),
                        "triangles_kept_pct_of_original": round(100 * float((d.faces * p).sum() / d.faces.sum()), 1),
                        "triangles_vs_oracle": round(float((d.faces * p).sum() / oracle_tris), 2),
                        "median_abs_error_keep": round(float(np.median(np.abs(p - y))), 3)})
        # The deployed procedure: start from a guess, check, and step the ratio up x1.6 until the budget is met.
        # Every method then ships 100% within budget; what differs is triangles shipped and checks needed.
        for name in ["model_median", "model_q90", "fixed_median"]:
            p = preds[name].copy()
            n = np.ceil(np.log(np.maximum(y.values / p, 1)) / np.log(1.6)).clip(0)
            final = np.minimum(p * 1.6 ** n, 1.0)
            out.append({"lod": lod, "method": f"{name} + verify/step-up (shipped)", "compliance_pct": 100.0,
                        "triangles_kept_pct_of_original": round(100 * float((d.faces * final).sum() / d.faces.sum()), 1),
                        "triangles_vs_oracle": round(float((d.faces * final).sum() / oracle_tris), 2),
                        "median_abs_error_keep": round(float(np.median(np.abs(final - y))), 3),
                        "avg_checks_per_part": round(float((n + 1).mean()), 2)})
        out.append({"lod": lod, "method": "oracle (binary search)", "avg_checks_per_part": 8.0, "compliance_pct": 100.0,
                    "triangles_kept_pct_of_original": round(100 * oracle_tris / d.faces.sum(), 1), "triangles_vs_oracle": 1.0,
                    "median_abs_error_keep": 0.0})
        models[lod.upper()] = gbr(0.5).fit(d[FEATS], z)          # deployed: median model + verify/step-up safety net
    ev = pd.DataFrame(out)
    ev.to_csv(ROOT / "results" / "keep_model_eval.csv", index=False)
    joblib.dump(models, ROOT / "results" / "keep_model.joblib")
    imp = pd.Series(models["LOD1"].feature_importances_, index=FEATS).sort_values(ascending=False).round(3)
    S = {"parts": int(len(d)), "assets": int(d.asset.nunique()), "median_oracle_seconds_per_part": round(float(d.oracle_seconds.median()), 2),
         "median_keep_lod1": round(float(d.keep_lod1.median()), 3), "median_keep_lod2": round(float(d.keep_lod2.median()), 3),
         "corr_boundary_fraction_vs_keep_lod1": round(float(d.frac_boundary_edges.corr(d.keep_lod1, method="spearman")), 2),
         "top_features_lod1": imp.head(5).to_dict()}
    (ROOT / "results" / "keep_model_summary.json").write_text(json.dumps(S, indent=1))
    print(json.dumps(S, indent=1)); print(ev.to_string(index=False))


if __name__ == "__main__":
    main()
