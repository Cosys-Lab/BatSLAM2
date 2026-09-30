"""Run BatSLAM 2.0 on one dataset and report against ground truth.

    python scripts/run_batslam.py                                  # data/hall.mat, the dataset's own odometry
    python scripts/run_batslam.py --dataset city.mat --odom-bias 0 # calibrated odometry, as in the paper
    python scripts/run_batslam.py --config my.json                 # partial config overrides
    python scripts/run_batslam.py --out results/exp1

Dataset names without a folder are looked up in data/ (or in $BATSLAM_DATA if set).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from batslam.config import BatSLAMConfig  # noqa: E402
from batslam.dataset import load_dataset, resynthesize_odometry  # noqa: E402
from batslam.eval.metrics import summarize  # noqa: E402
from batslam.eval.plots import plot_errors, plot_run  # noqa: E402
from batslam.frontend.cochleogram import compute_all  # noqa: E402
from batslam.slam import run  # noqa: E402


DATA = Path(os.environ.get("BATSLAM_DATA", ROOT / "data"))


def dataset_path(name) -> Path:
    """A dataset given by path, or by file name inside the data folder."""
    p = Path(name)
    return p if p.exists() or p.is_absolute() or len(p.parts) > 1 else DATA / p


def descriptors_cached(ds, cfg, path: Path):
    """Front-end output, cached next to the dataset and keyed by the front-end config."""
    key = hashlib.sha1(json.dumps(cfg.to_dict()["frontend"], sort_keys=True).encode()).hexdigest()[:10]
    cache = path.with_suffix(f".desc_{key}.npz")
    if cache.exists() and cache.stat().st_mtime > path.stat().st_mtime:
        z = np.load(cache)
        return z["D"], float(z["range_bin"])
    t = time.time()
    D, fe = compute_all(ds, cfg.frontend)
    print(f"frontend: {D.shape} in {time.time() - t:.1f}s")
    np.savez_compressed(cache, D=D, range_bin=cfg.frontend.range_bin)
    return D, cfg.frontend.range_bin


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="hall.mat")
    ap.add_argument("--config", default=None, help="JSON with partial config overrides")
    ap.add_argument("--odom-bias", type=float, default=None,
                    help="redraw the odometry with this bias factor (0 = calibrated, 1 = the simulated bias)")
    ap.add_argument("--odom-seed", type=int, default=1, help="noise seed for --odom-bias")
    ap.add_argument("--out", default=str(ROOT / "results" / "latest"))
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args(argv)

    cfg = BatSLAMConfig.load(a.config) if a.config else BatSLAMConfig()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    path = dataset_path(a.dataset)
    ds = load_dataset(path)
    if a.odom_bias is not None:
        resynthesize_odometry(ds, a.odom_bias, 1.0, a.odom_seed)
    print(f"dataset: {ds.n} pulses, laps {ds.lap_names}")
    D, range_bin = descriptors_cached(ds, cfg, path)

    t = time.time()
    res = run(ds, D, cfg, range_bin)
    print(f"slam: {time.time() - t:.1f}s")

    summary = summarize(res, ds, cfg)
    print(json.dumps(summary, indent=2))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    cfg.save(out / "config.json")
    np.savez(out / "trajectories.npz", gt=ds.gt_pose, odometry=res.odometry, slam=res.poses,
             slam_online=res.poses_online, templates=res.template_anchor_keys,
             links=np.array([[l.query_key, l.anchor_key, l.hypothesis_id] for l in res.loops]).reshape(-1, 3))
    if not a.no_plots:
        o, s = summary["ate_odometry"]["rmse_m"], summary["ate_slam"]["rmse_m"]
        plot_run(res, ds, out / "map.png",
                 title=f"BatSLAM 2.0: ATE odometry {o:.2f} m -> SLAM {s:.2f} m, "
                       f"link precision {summary['link_precision']:.2f}")
        plot_errors(res, ds, out / "errors.png")
        print(f"figures -> {out}")
    return summary


if __name__ == "__main__":
    main()
