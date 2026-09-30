"""BatSLAM 2.0 pipeline: echoes + odometry in, optimised pose graph out.

Per pulse:
    odometry  -> new pose node (iSAM2)
    echo      -> cochleogram descriptor              (frontend)
    descriptor-> top-k template candidates           (views, pose-gated)
    candidates-> sequence verifier                   (recognition)
    verified  -> weak robust view links              (backend)
    no lock + moved enough -> lay a new template anchored at this node
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np

from .backend.pose_graph import PoseGraph, pose2
from .config import BatSLAMConfig
from .recognition.sequence import SequenceVerifier
from .views.templates import Candidate, TemplateStore


@dataclass
class StepLog:
    key: int
    n_candidates: int
    best_similarity: float
    n_alive: int
    locked: bool
    new_template: bool
    n_links: int


@dataclass
class BatSLAMResult:
    poses: np.ndarray               # final (batch-refined) estimate [N x 3]
    poses_online: np.ndarray        # iSAM2 estimate at the end of the run
    odometry: np.ndarray            # dead reckoning [N x 3]
    template_anchor_keys: np.ndarray
    loops: list                     # links in the final graph
    log: list[StepLog] = field(default_factory=list)
    n_rejected_implausible: int = 0
    commit_log: list = field(default_factory=list)       # features of every commit decision
    rejected_links: list = field(default_factory=list)   # links whose group was removed
    events: list = field(default_factory=list)           # (node, group, event) from link management


class BatSLAM:
    def __init__(self, cfg: BatSLAMConfig, descriptor_shape, range_bin: float,
                 initial_pose=(0.0, 0.0, 0.0)):
        self.cfg = cfg
        self.graph = PoseGraph(cfg.graph)
        self.templates = TemplateStore(cfg.templates, descriptor_shape, range_bin)
        self.odom: list[gtsam.Pose2] = []
        self.odom_s: list[float] = []
        self.verifier = SequenceVerifier(cfg.recognition,
                                         odom_pose=lambda k: self.odom[k],
                                         map_pose=lambda k: self.graph.pose(k),
                                         use_shift_offset=cfg.graph.use_shift_offset,
                                         plausible=self._plausible)
        self._initial = initial_pose
        self._hist: list[tuple[float, np.ndarray]] = []    # (odom s, view) for stacked local views
        self._last_template_s = -np.inf
        self._last_template_theta = 0.0
        self.log: list[StepLog] = []

    # ------------------------------------------------------------------
    def _admissible(self, key: int):
        tc = self.cfg.templates
        if tc.gate_radius_m is None and tc.gate_heading_rad is None:
            return None
        cur = self.graph.pose(key)

        def ok(t) -> bool:
            e = cur.between(self.graph.pose(t.anchor_key))
            if tc.gate_radius_m is not None and np.hypot(e.x(), e.y()) > tc.gate_radius_m:
                return False
            if tc.gate_heading_rad is not None and abs(e.theta()) > tc.gate_heading_rad:
                return False
            return True
        return ok

    def _plausible(self, implied: gtsam.Pose2, key: int, anchor: int | None = None) -> tuple[float, float, float]:
        """How compatible is 'we are actually at ``implied``' with the current belief?
        Returns (Mahalanobis d^2, correction distance [m], correction heading [rad])."""
        e = self.graph.pose(key).between(implied)
        v = np.array([e.x(), e.y(), e.theta()])
        dist = float(np.hypot(e.x(), e.y()))
        rc = self.cfg.recognition
        if rc.plausibility_relative and anchor is not None and dist > rc.plausibility_relative_min_m:
            # large correction: test it against the uncertainty of the anchor -> now relative
            # pose (joint marginal). Both nodes share their drift from the origin, so the
            # current node's marginal alone overstates how far the map may move.
            C = self.graph.relative_covariance(anchor, key)
        else:
            # small correction: the current node's marginal (relative to the prior at node 0),
            # a looser but cheap bound
            C = self.graph.marginal_covariance(key)
        P = C + np.diag(np.square(self.cfg.graph.loop_sigma))
        return float(v @ np.linalg.solve(P, v)), dist, float(abs(e.theta()))

    @staticmethod
    def descriptor_shape(cfg: BatSLAMConfig, view_shape) -> tuple:
        n = cfg.templates.history_n
        return (view_shape[0] * n,) + tuple(view_shape[1:])

    def _local_view(self, view: np.ndarray, s: float) -> np.ndarray:
        """Stack of the last history_n views at odometric spacing (causal)."""
        tc = self.cfg.templates
        if tc.history_n <= 1:
            return view
        self._hist.append((s, view))
        span = (tc.history_n - 1) * tc.history_spacing_m
        while len(self._hist) > 2 and self._hist[1][0] < s - span - 1.0:
            self._hist.pop(0)
        ss = np.array([h[0] for h in self._hist])
        parts = []
        for k in range(tc.history_n):
            j = int(np.clip(np.searchsorted(ss, s - k * tc.history_spacing_m), 0, len(ss) - 1))
            parts.append(self._hist[j][1])
        return np.concatenate(parts, axis=0)

    def process(self, descriptor: np.ndarray, odom_delta=None) -> StepLog:
        """Feed one pulse (a single view). ``odom_delta`` is None for the very first pulse."""
        if odom_delta is None:
            key = self.graph.add_first(self._initial)
            self.odom.append(pose2(self._initial))
            self.odom_s.append(0.0)
        else:
            key = self.graph.add_odometry(odom_delta)
            self.odom.append(self.odom[-1].compose(pose2(odom_delta)))
            self.odom_s.append(self.odom_s[-1] + float(np.hypot(odom_delta[0], odom_delta[1])))
        self.graph.update()
        s = self.odom_s[key]
        descriptor = self._local_view(descriptor, s)

        # recognise
        self.templates.observe(descriptor)
        cands: list[Candidate] = self.templates.query(key, descriptor, s, self._admissible(key))
        res = self.verifier.step(key, cands)

        # inject weak links for verified recognitions
        for c, hid in res.release:
            z = self.verifier.measurement(c)
            self.graph.add_view_link(c.query_key, c.anchor_key, (z.x(), z.y(), z.theta()),
                                     hypothesis_id=hid, similarity=c.similarity)
        if res.release:
            self.graph.update(extra_iterations=self.cfg.graph.extra_iterations_on_loop)
        # multi-hypothesis link management: tentative groups may be removed
        # again, rejected ones re-admitted; a removed group's hypothesis stops
        if self.cfg.graph.link_management:
            for gid in self.graph.manage(key, new_links=bool(res.release)):
                self.verifier.drop(gid)
                res.locked = res.locked and self.verifier.is_locked()

        # lay a template unless we are locked onto known territory
        tc = self.cfg.templates
        theta = self.odom[key].theta()
        moved = s - self._last_template_s >= tc.spacing_m
        turned = abs(np.angle(np.exp(1j * (theta - self._last_template_theta)))) >= tc.spacing_rad
        new_t = (not res.locked) and (moved or turned)
        if new_t:
            self.templates.add(key, descriptor, s)
            self._last_template_s = s
            self._last_template_theta = theta

        entry = StepLog(key, len(cands), max((c.similarity for c in cands), default=float("nan")),
                        res.n_alive, res.locked, new_t, len(res.release))
        self.log.append(entry)
        return entry

    def result(self, batch_refine: bool = True) -> BatSLAMResult:
        if self.cfg.graph.link_management:
            self.graph.manage(self.graph.n - 1, force_audit=True)     # final audit
        online = self.graph.poses()
        final = self.graph.batch_optimize() if batch_refine else online
        odo = np.array([[p.x(), p.y(), p.theta()] for p in self.odom])
        return BatSLAMResult(final, online, odo,
                             np.array([t.anchor_key for t in self.templates.templates]),
                             list(self.graph.loops), list(self.log),
                             self.verifier.n_rejected_implausible,
                             list(self.verifier.commit_log),
                             [l for l in self.graph.all_links if l.status == "rejected"],
                             list(self.graph.events))


def run(dataset, descriptors: np.ndarray, cfg: BatSLAMConfig, range_bin: float,
        progress: bool = True) -> BatSLAMResult:
    """Offline driver over a whole dataset (descriptors precomputed)."""
    slam = BatSLAM(cfg, BatSLAM.descriptor_shape(cfg, descriptors.shape[1:]), range_bin,
                   initial_pose=tuple(dataset.gt_pose[0]))
    for i in range(dataset.n):
        slam.process(descriptors[i], None if i == 0 else dataset.odom_delta[i - 1])
        if progress and (i + 1) % 200 == 0:
            e = slam.log[-1]
            print(f"  step {i + 1}/{dataset.n}: templates={len(slam.templates)} "
                  f"links={len(slam.graph.loops)} hyps={e.n_alive}")
    return slam.result()
