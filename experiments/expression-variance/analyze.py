"""
analyze.py
----------
Groups per-(seq, exp) results from experiment.py by (seq_name, obj_id) and
computes {min, mean, max, variance, std, range} for each of:
  - mean_iou
  - mean_mass_in_gt
  - mean_mass_in_pred

Inputs : results.json produced by experiment.py
Outputs: grouped_stats.json — one entry per (seq, obj) group
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

METRICS = ("mean_iou", "mean_mass_in_gt", "mean_mass_in_pred")


def _stats(values):
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    return {
        "n":        int(arr.size),
        "min":      float(arr.min()),
        "mean":     float(arr.mean()),
        "max":      float(arr.max()),
        "variance": float(arr.var(ddof=0)),
        "std":      float(arr.std(ddof=0)),
        "range":    float(arr.max() - arr.min()),
        "values":   [float(x) for x in arr],
    }


def group_results(results: list) -> dict:
    groups: dict = defaultdict(lambda: {"expressions": [], "exp_ids": [],
                                        "mean_iou": [], "mean_mass_in_gt": [],
                                        "mean_mass_in_pred": []})
    for r in results:
        if r.get("error"):
            continue
        key = f'{r["seq_name"]}__obj{r["obj_id"]}'
        g = groups[key]
        g["expressions"].append(r["expression"])
        g["exp_ids"].append(r["exp_id"])
        for m in METRICS:
            g[m].append(r.get(m))

    out: dict = {}
    for key, g in groups.items():
        seq, obj = key.rsplit("__obj", 1)
        out[key] = {
            "seq_name":   seq,
            "obj_id":     int(obj),
            "expressions": g["expressions"],
            "exp_ids":    g["exp_ids"],
            "iou":          _stats(g["mean_iou"]),
            "mass_in_gt":   _stats(g["mean_mass_in_gt"]),
            "mass_in_pred": _stats(g["mean_mass_in_pred"]),
        }
    return out


def summary(grouped: dict) -> dict:
    """Dataset-level rollup of variances & best-vs-worst gaps."""
    rows = list(grouped.values())
    out = {"num_groups": len(rows)}
    for m, key in [("iou", "iou"), ("mass_in_gt", "mass_in_gt"),
                   ("mass_in_pred", "mass_in_pred")]:
        variances = [r[key]["variance"] for r in rows if r[key]]
        ranges    = [r[key]["range"]    for r in rows if r[key]]
        gaps      = [r[key]["max"] - r[key]["min"] for r in rows if r[key]]
        bests     = [r[key]["max"]   for r in rows if r[key]]
        worsts    = [r[key]["min"]   for r in rows if r[key]]
        means     = [r[key]["mean"]  for r in rows if r[key]]
        out[m] = {
            "mean_within_group_variance": float(np.mean(variances)) if variances else None,
            "mean_within_group_std":      float(np.mean([np.sqrt(v) for v in variances])) if variances else None,
            "mean_within_group_range":    float(np.mean(ranges))    if ranges    else None,
            "mean_best":                  float(np.mean(bests))     if bests     else None,
            "mean_worst":                 float(np.mean(worsts))    if worsts    else None,
            "mean_avg":                   float(np.mean(means))     if means     else None,
            "mean_best_minus_worst":      float(np.mean(gaps))      if gaps      else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True, help="Path to results.json")
    ap.add_argument("--out_dir", default=None,
                    help="Default = parent of results.json")
    args = ap.parse_args()

    results_path = Path(args.results)
    out_dir = Path(args.out_dir) if args.out_dir else results_path.parent

    results = json.load(open(results_path))
    grouped = group_results(results)
    summ    = summary(grouped)

    grouped_path = out_dir / "grouped_stats.json"
    summary_path = out_dir / "summary.json"
    grouped_path.write_text(json.dumps(grouped, indent=2))
    summary_path.write_text(json.dumps(summ,    indent=2))

    print(f"Wrote {grouped_path}")
    print(f"Wrote {summary_path}")
    print("\n=== Dataset-level summary ===")
    print(f"  groups: {summ['num_groups']}")
    for m in ("iou", "mass_in_gt", "mass_in_pred"):
        s = summ[m]
        print(f"  {m}:")
        print(f"    mean within-group std:      {s['mean_within_group_std']:.4f}")
        print(f"    mean within-group range:    {s['mean_within_group_range']:.4f}")
        print(f"    mean best - worst gap:      {s['mean_best_minus_worst']:.4f}")
        print(f"    mean(worst) → mean(best):   {s['mean_worst']:.4f} → {s['mean_best']:.4f}")


if __name__ == "__main__":
    main()
