"""Compare runs of the same name across result folders.

    python scripts/compare_suites.py results/main results/main_rerun
Prints, per run: ATE, aligned ATE, wrong commits (final / all), coverage, collapsed pairs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


def load(d):
    out = {}
    for f in sorted(Path(d).glob("*/summary.json")):
        s = json.loads(f.read_text())
        cf = f.with_name("commits.json")
        cm = json.loads(cf.read_text()) if cf.exists() else []
        s["_wrong_all"] = sum(c["correct_frac"] < 0.5 for c in cm)
        out[f.parent.name] = s
    return out


def fmt(s):
    if s is None:
        return f"{'-':>31}"
    return (f"{s['ate_slam']['rmse_m']:5.2f} {s['ate_slam_aligned']['rmse_m']:5.2f} "
            f"{s['n_bad_commits']:2d}/{s['_wrong_all']:<2d} {s['revisit_coverage']:.2f} "
            f"{100 * s['collapse_slam']['frac_collapsed']:4.1f}")


def main():
    dirs = sys.argv[1:]
    S = [load(d) for d in dirs]
    names = sorted(set().union(*S))
    print(f"{'run':<16}" + "".join(f"| {Path(d).name[:29]:<31}" for d in dirs))
    print(f"{'':<16}" + "| ATE   algn  wr/all cov  coll " * len(dirs))
    for n in names:
        print(f"{n:<16}" + "".join(f"| {fmt(s.get(n))} " for s in S))
    for d, s in zip(dirs, S):
        common = [n for n in names if all(n in x for x in S)]
        if not common:
            continue
        a = [s[n]["ate_slam"]["rmse_m"] for n in common]
        b = [s[n]["ate_slam_aligned"]["rmse_m"] for n in common]
        w = sum(s[n]["n_bad_commits"] for n in common)
        print(f"{Path(d).name}: {len(common)} common runs, mean ATE {sum(a) / len(a):.2f}, max {max(a):.2f}, "
              f"mean aligned {sum(b) / len(b):.2f}, wrong final {w}")


if __name__ == "__main__":
    main()
