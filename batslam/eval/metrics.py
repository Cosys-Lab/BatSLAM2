"""Ground-truth evaluation: trajectory error and view-link correctness."""
from __future__ import annotations

import numpy as np


def wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def align_se2(est: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Least-squares rigid 2D alignment of est onto gt (Umeyama, no scale)."""
    mu_e, mu_g = est[:, :2].mean(0), gt[:, :2].mean(0)
    H = (est[:, :2] - mu_e).T @ (gt[:, :2] - mu_g)
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    out = est.copy()
    out[:, :2] = (est[:, :2] - mu_e) @ R.T + mu_g
    out[:, 2] = wrap(est[:, 2] + np.arctan2(R[1, 0], R[0, 0]))
    return out


def ate(est: np.ndarray, gt: np.ndarray, align: bool = False) -> dict:
    e = align_se2(est, gt) if align else est
    d = np.hypot(*(e[:, :2] - gt[:, :2]).T)
    th = np.abs(wrap(e[:, 2] - gt[:, 2]))
    return {"rmse_m": float(np.sqrt(np.mean(d ** 2))), "max_m": float(d.max()),
            "final_m": float(d[-1]), "rmse_deg": float(np.degrees(np.sqrt(np.mean(th ** 2))))}


def link_truth(loops, gt: np.ndarray, tol_m: float = 0.6, tol_rad: float = 0.45) -> np.ndarray:
    """Per view link: is the anchor really near the query (in x, y AND heading)?"""
    ok = np.zeros(len(loops), dtype=bool)
    for i, l in enumerate(loops):
        q, a = gt[l.query_key], gt[l.anchor_key]
        ok[i] = (np.hypot(q[0] - a[0], q[1] - a[1]) < tol_m
                 and abs(wrap(q[2] - a[2])) < tol_rad)
    return ok


def revisit_mask(gt: np.ndarray, min_gap_m: float, s: np.ndarray, tol_m: float = 0.6,
                 tol_rad: float = 0.45) -> np.ndarray:
    """Pulses that genuinely revisit an earlier place (same x, y, heading)."""
    n = len(gt)
    out = np.zeros(n, dtype=bool)
    for i in range(n):
        earlier = s < s[i] - min_gap_m
        if not earlier.any():
            continue
        g = gt[earlier]
        d = np.hypot(g[:, 0] - gt[i, 0], g[:, 1] - gt[i, 1])
        th = np.abs(wrap(g[:, 2] - gt[i, 2]))
        out[i] = bool(np.any((d < tol_m) & (th < tol_rad)))
    return out


def collapse_stats(est: np.ndarray, gt: np.ndarray, min_gt_m: float = 4.0, n_pairs: int = 200_000,
                   seed: int = 0) -> dict:
    """Map-collapse check: do distinct places stay apart in the estimate?

    For random node pairs at least ``min_gt_m`` apart in ground truth, the
    ratio of estimated to true distance. A collapsing map (false loop
    closures fusing different places) drives ratios well below 1; a merely
    drifting map spreads them both ways. Rigid-motion invariant, so no
    alignment needed.
    """
    rng = np.random.default_rng(seed)
    i = rng.integers(0, len(gt), n_pairs)
    j = rng.integers(0, len(gt), n_pairs)
    dg = np.hypot(*(gt[i, :2] - gt[j, :2]).T)
    keep = dg >= min_gt_m
    de = np.hypot(*(est[i[keep], :2] - est[j[keep], :2]).T)
    r = de / dg[keep]
    return {"median_ratio": float(np.median(r)), "p05_ratio": float(np.percentile(r, 5)),
            "frac_collapsed": float(np.mean(r < 0.5)),   # pairs pulled to < half their true distance
            "frac_fused": float(np.mean(de < 1.0))}      # truly >= min_gt_m apart, estimated < 1 m


def hypothesis_truth(loops, truth: np.ndarray) -> dict:
    """Per committed hypothesis: fraction of its links that are correct."""
    by = {}
    for l, ok in zip(loops, truth):
        by.setdefault(l.hypothesis_id, []).append(ok)
    fr = np.array([np.mean(v) for v in by.values()]) if by else np.zeros(0)
    return {"n_committed": int(len(fr)), "n_bad_commits": int(np.sum(fr < 0.5))}


def summarize(result, dataset, cfg) -> dict:
    gt = dataset.gt_pose
    s = np.concatenate([[0], np.cumsum(np.hypot(*dataset.gt_delta[:, :2].T))])
    truth = link_truth(result.loops, gt)
    revisits = revisit_mask(gt, cfg.templates.exclude_recent_m, s)
    linked = np.zeros(len(gt), dtype=bool)
    for l, ok in zip(result.loops, truth):
        if ok:
            linked[l.query_key] = True
    return {
        "n_pulses": int(len(gt)),
        "n_templates": int(len(result.template_anchor_keys)),
        "n_links": int(len(result.loops)),
        "link_precision": float(truth.mean()) if len(truth) else float("nan"),
        "n_false_links": int((~truth).sum()),
        "revisit_coverage": float(linked[revisits].mean()) if revisits.any() else float("nan"),
        "n_hypotheses_committed": int(len({l.hypothesis_id for l in result.loops})),
        "n_bad_commits": hypothesis_truth(result.loops, truth)["n_bad_commits"],
        "n_rejected_links": int(len(getattr(result, "rejected_links", []))),
        "rejected_links_false_frac": (float(1 - link_truth(result.rejected_links, gt).mean())
                                      if getattr(result, "rejected_links", []) else float("nan")),
        "n_link_events": int(len(getattr(result, "events", []))),
        "n_rejected_implausible": int(getattr(result, "n_rejected_implausible", 0)),
        "ate_odometry": ate(result.odometry, gt),
        "ate_slam_online": ate(result.poses_online, gt),
        "ate_slam": ate(result.poses, gt),
        "ate_slam_aligned": ate(result.poses, gt, align=True),
        "collapse_odometry": collapse_stats(result.odometry, gt),
        "collapse_slam": collapse_stats(result.poses, gt),
    }
