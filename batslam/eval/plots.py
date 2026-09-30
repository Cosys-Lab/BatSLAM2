"""Figures for a BatSLAM 2.0 run."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .metrics import link_truth  # noqa: E402

C_GT, C_ODO, C_SLAM = "#222222", "#d95f02", "#1b73b3"
C_OK, C_BAD = "#1a9850", "#d73027"


def draw_scene(ax, dataset, alpha=0.6):
    r = dataset.reflectors
    for i, t in enumerate(r.type):
        if t == "wall":
            ax.plot([r.endpoint1[i, 0], r.endpoint2[i, 0]], [r.endpoint1[i, 1], r.endpoint2[i, 1]],
                    color="#888", lw=1.5, alpha=alpha, zorder=1)
        elif t == "circle":
            col = "#8e44ad" if r.has_signature[i] else "#888"
            ax.add_patch(plt.Circle(r.position[i, :2], r.radius[i], color=col, alpha=alpha, zorder=1))
        else:
            ax.plot(*r.position[i, :2], ".", color="#aaa", ms=3, zorder=1)
    trap = np.atleast_2d(dataset.meta.get("alias_trap", np.zeros((0, 2))))
    if trap.size:
        ax.plot(trap[:, 0], trap[:, 1], "x", color=C_BAD, ms=5, alpha=0.7, zorder=2,
                label="aliasing-trap pillars")


def plot_run(result, dataset, path, title: str = ""):
    gt = dataset.gt_pose
    truth = link_truth(result.loops, gt)
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.2), constrained_layout=True)

    ax = axes[0]
    draw_scene(ax, dataset)
    ax.plot(gt[:, 0], gt[:, 1], color=C_GT, lw=1.2, label="ground truth")
    ax.plot(result.odometry[:, 0], result.odometry[:, 1], color=C_ODO, lw=1, label="odometry")
    ax.plot(result.poses[:, 0], result.poses[:, 1], color=C_SLAM, lw=1.2, label="BatSLAM 2.0")
    ax.set_title("Trajectories")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)

    ax = axes[1]
    draw_scene(ax, dataset)
    ax.plot(result.poses[:, 0], result.poses[:, 1], color=C_SLAM, lw=0.8, alpha=0.7)
    a = result.template_anchor_keys
    ax.plot(result.poses[a, 0], result.poses[a, 1], "o", ms=2.5, color=C_SLAM, label="templates")
    for l in getattr(result, "rejected_links", []):
        q, t = result.poses[l.query_key], result.poses[l.anchor_key]
        ax.plot([q[0], t[0]], [q[1], t[1]], color="#999", lw=0.6, ls=":", alpha=0.6, zorder=2)
    if getattr(result, "rejected_links", []):
        ax.plot([], [], color="#999", ls=":", label=f"removed by link management ({len(result.rejected_links)})")
    for l, ok in zip(result.loops, truth):
        q, t = result.poses[l.query_key], result.poses[l.anchor_key]
        ax.plot([q[0], t[0]], [q[1], t[1]], color=C_OK if ok else C_BAD, lw=0.8 if ok else 1.5,
                alpha=0.8, zorder=3)
    ax.plot([], [], color=C_OK, label=f"view links ok ({truth.sum()})")
    ax.plot([], [], color=C_BAD, label=f"view links wrong ({(~truth).sum()})")
    ax.set_title("Template map and verified view links (on the SLAM estimate)")
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=8)
    if title:
        fig.suptitle(title)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_errors(result, dataset, path):
    gt = dataset.gt_pose
    s = np.concatenate([[0], np.cumsum(np.hypot(*dataset.gt_delta[:, :2].T))])
    fig, axes = plt.subplots(3, 1, figsize=(12, 7.5), sharex=True, constrained_layout=True)
    for est, col, lab in [(result.odometry, C_ODO, "odometry"), (result.poses_online, "#7fb2d9", "iSAM2 online"),
                          (result.poses, C_SLAM, "BatSLAM 2.0 (final)")]:
        axes[0].plot(s, np.hypot(*(est[:, :2] - gt[:, :2]).T), color=col, label=lab)
    axes[0].set_ylabel("position error (m)")
    axes[0].legend(fontsize=8)
    laps = dataset.lap
    for k in np.unique(laps):
        idx = np.where(laps == k)[0]
        for ax in axes:
            ax.axvspan(s[idx[0]], s[idx[-1]], color="#000", alpha=0.04 * (k % 2))
        axes[0].text(s[idx[0]], axes[0].get_ylim()[1] * 0.9, dataset.lap_names[k - 1], fontsize=8)
    log = result.log
    axes[1].plot(s, [e.best_similarity for e in log], ".", ms=2, color="#555")
    axes[1].set_ylabel("best candidate sim")
    axes[2].plot(s, [e.n_alive for e in log], color="#555", lw=0.8, label="alive hypotheses")
    locked = np.array([e.locked for e in log])
    axes[2].fill_between(s, 0, locked * max(1, max(e.n_alive for e in log)), color=C_OK, alpha=0.25,
                         step="mid", label="locked on (links injected)")
    axes[2].set_ylabel("hypotheses")
    axes[2].set_xlabel("distance travelled (m)")
    axes[2].legend(fontsize=8)
    fig.savefig(path, dpi=130)
    plt.close(fig)
