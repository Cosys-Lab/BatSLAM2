"""Run a set of BatSLAM 2.0 experiments in parallel and tabulate them.

An experiment file is a JSON list of runs:
    [{"name": "r3_b0_s1", "dataset": "indoor_r3.mat",
      "config": {"graph": {"link_management": false}},     # partial overrides of the defaults
      "odom_bias_factor": 0.0, "odom_noise_factor": 1.0, "odom_seed": 1}, ...]
("config" may also be a path to a JSON file. The odometry keys are optional: without them
the dataset's own odometry is used. Dataset names are looked up in data/ or $BATSLAM_DATA.)

    python scripts/run_suite.py experiments/main.json --out results/main --workers 4
    python scripts/run_suite.py experiments/main.json --out results/main --only hall_b0

Writes per run: <out>/<name>/{summary.json, config.json, links.npz, commits.json, map.png, errors.png};
and for the set: <out>/table.md and <out>/table.json.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

COLUMNS = [  # (header, key path, format)
    ("links", ("n_links",), "{:d}"),
    ("precision", ("link_precision",), "{:.3f}"),
    ("bad commits", ("_bad",), "{}"),
    ("revisits linked", ("revisit_coverage",), "{:.2f}"),
    ("removed (false %)", ("_removed",), "{}"),
    ("ATE odo [m]", ("ate_odometry", "rmse_m"), "{:.2f}"),
    ("ATE SLAM [m]", ("ate_slam", "rmse_m"), "{:.2f}"),
    ("max err [m]", ("ate_slam", "max_m"), "{:.2f}"),
    ("heading RMSE [deg]", ("ate_slam", "rmse_deg"), "{:.1f}"),
    ("collapsed pairs", ("collapse_slam", "frac_collapsed"), "{:.3f}"),
    ("runtime [s]", ("_runtime",), "{:.0f}"),
]


def _cfg(spec):
    from batslam.config import BatSLAMConfig
    c = spec.get("config", {})
    if isinstance(c, str):
        c = json.loads(Path(c).read_text())
    return BatSLAMConfig.from_dict(c)


def run_one(spec: dict, out_root: str) -> dict:
    from batslam.dataset import load_dataset, resynthesize_odometry
    from batslam.eval.metrics import summarize
    from batslam.eval.plots import plot_errors, plot_run
    from batslam.slam import run
    from run_batslam import dataset_path, descriptors_cached

    name = spec["name"]
    out = Path(out_root) / name
    out.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(spec)
    path = dataset_path(spec["dataset"])
    ds = load_dataset(path)
    if "odom_bias_factor" in spec or "odom_noise_factor" in spec:
        resynthesize_odometry(ds, spec.get("odom_bias_factor", 1.0), spec.get("odom_noise_factor", 1.0),
                              spec.get("odom_seed", 1))
    D, rb = descriptors_cached(ds, cfg, path)
    t = time.time()
    res = run(ds, D, cfg, rb, progress=False)
    s = summarize(res, ds, cfg)
    s["_runtime"] = time.time() - t
    s["_spec"] = spec
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    # every link ever committed, with its final status and ground-truth verdict
    import numpy as np
    from batslam.eval.metrics import link_truth
    links = list(res.loops) + list(res.rejected_links)
    ok = link_truth(links, ds.gt_pose) if links else np.zeros(0, bool)
    np.savez(out / "links.npz",
             query=np.array([l.query_key for l in links], int), anchor=np.array([l.anchor_key for l in links], int),
             group=np.array([l.hypothesis_id for l in links], int),
             active=np.array([l.status == "active" for l in links], bool), correct=ok,
             gt=ds.gt_pose, slam=res.poses, odometry=res.odometry)
    # commit decisions with their ground-truth verdict (majority of the group's links)
    commits = []
    for c in res.commit_log:
        m = np.array([l.hypothesis_id == c["id"] for l in links], bool)
        c = dict(c, correct_frac=float(ok[m].mean()) if m.any() else float("nan"), n_links_final=int(m.sum()))
        commits.append(c)
    (out / "commits.json").write_text(json.dumps(commits, indent=1))
    cfg.save(out / "config.json")
    o, e = s["ate_odometry"]["rmse_m"], s["ate_slam"]["rmse_m"]
    plot_run(res, ds, out / "map.png", title=f"{name}: ATE odometry {o:.2f} m -> SLAM {e:.2f} m, "
             f"link precision {s['link_precision']:.2f}")
    plot_errors(res, ds, out / "errors.png")
    print(f"[done] {name}: ATE {e:.2f} m (odo {o:.2f}), precision {s['link_precision']:.3f}, "
          f"{s['_runtime']:.0f}s", flush=True)
    return s


def _get(s, path):
    if path == ("_bad",):
        return f"{s['n_bad_commits']}/{s['n_hypotheses_committed']}"
    if path == ("_removed",):
        n = s.get("n_rejected_links", 0)
        f = s.get("rejected_links_false_frac", float("nan"))
        return f"{n} ({100 * f:.0f}%)" if n else "0"
    v = s
    for k in path:
        v = v[k]
    return v


def table(rows: list[tuple[str, dict]]) -> str:
    head = "| run | " + " | ".join(c[0] for c in COLUMNS) + " |"
    sep = "|---|" + "---|" * len(COLUMNS)
    lines = [head, sep]
    for name, s in rows:
        cells = []
        for _, path, fmt in COLUMNS:
            v = _get(s, path)
            cells.append(fmt.format(v) if not isinstance(v, str) else v)
        lines.append(f"| {name} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--only", nargs="*", default=None, help="run only these names")
    a = ap.parse_args()
    specs = json.loads(Path(a.suite).read_text())
    if a.only:
        specs = [s for s in specs if s["name"] in a.only]
    # frontend caches first (serially), so workers never race on writing them
    from batslam.dataset import load_dataset
    from run_batslam import dataset_path, descriptors_cached
    seen = set()
    for sp in specs:
        cfg = _cfg(sp)
        key = (sp["dataset"], json.dumps(cfg.to_dict()["frontend"], sort_keys=True))
        if key not in seen:
            seen.add(key)
            p = dataset_path(sp["dataset"])
            descriptors_cached(load_dataset(p), cfg, p)
    with mp.get_context("spawn").Pool(a.workers) as pool:
        results = pool.starmap(run_one, [(sp, a.out) for sp in specs])
    rows = [(sp["name"], r) for sp, r in zip(specs, results)]
    md = table(rows)
    Path(a.out, "table.md").write_text(md + "\n")
    Path(a.out, "table.json").write_text(json.dumps({n: r for n, r in rows}, indent=2, default=str))
    print(md)


if __name__ == "__main__":
    main()
