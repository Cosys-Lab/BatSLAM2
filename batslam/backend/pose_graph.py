"""GTSAM pose graph (iSAM2) with odometry and weak, robust, *removable* view links.

Nodes are one Pose2 per sonar pulse, keyed ``x<i>``. Odometry is a
BetweenFactorPose2 with noise growing with sqrt(distance). A recognised
local view becomes a *weak* BetweenFactorPose2 between the current node
and the template's anchor node - individually loose (decimetres, tens of
degrees) and wrapped in a robust kernel, so one wrong link cannot drag
the map; a verified sequence of them is what actually closes the loop.

Multi-hypothesis link management (``GraphConfig.link_management``)
------------------------------------------------------------------
The links of one committed recognition hypothesis form a **group**, and a
group is only ever *tentative* when it enters the graph:

  tentative --(survives 2 audits)--> confirmed
      |  ^                               |
      v  | (GNC audit re-admits)         v (GNC audit rejects)
    rejected <---------------------------+

* fast check, after every injection: a tentative group whose links no
  longer fit the current estimate (median per-link chi2 > group_chi2_max)
  is removed from iSAM2 immediately;
* GNC audit, every ``gnc_every`` nodes and at the end: graduated
  non-convexity (Geman-McClure) over the prior + odometry (known inliers)
  and ALL groups - active and rejected. Each group's mean GNC weight
  decides whether it is active. Because rejected groups take part, a group
  removed early can come back once later evidence supports it, and a
  confirmed group can still be thrown out if later evidence contradicts it;
* length rule: a group whose hypothesis ends (no new link for
  ``short_group_grace`` pulses) with fewer than ``min_group_links`` links is
  withdrawn for good. False recognitions in self-similar corridors are brief
  aliases - a metre or two that happens to look like another corridor - while
  a true revisit keeps matching as the robot drives on (indoor data: wrong
  groups had 9-15 links, correct ones a median of 30).

Removal uses iSAM2's ``removeFactorIndices`` - the graph really forgets.
This is the "tentative edge -> robust optimisation -> confirmed" pattern,
cf. RRR (Latif et al. 2013) and GNC (Yang et al. 2020).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import gtsam
import numpy as np

from ..config import GraphConfig


def X(i: int) -> int:
    return gtsam.symbol("x", i)


def pose2(v) -> gtsam.Pose2:
    return gtsam.Pose2(float(v[0]), float(v[1]), float(v[2]))


@dataclass
class LoopLink:
    query_key: int
    anchor_key: int
    measured: tuple[float, float, float]   # anchor pose in query frame
    hypothesis_id: int
    similarity: float
    status: str = "active"                 # mirrors its group: "active" | "rejected"


@dataclass
class LinkGroup:
    id: int
    links: list[LoopLink] = field(default_factory=list)
    factors: list = field(default_factory=list)
    isam_idx: list[int] | None = None      # factor indices while in iSAM2
    status: str = "tentative"              # "tentative" | "confirmed" | "rejected"
    created_key: int = 0
    audits_passed: int = 0
    permanent_reject: bool = False         # withdrawn for being too short: never re-admitted
    history: list[tuple[int, str]] = field(default_factory=list)   # (node, event)

    @property
    def active(self) -> bool:
        return self.status != "rejected"


class PoseGraph:
    def __init__(self, cfg: GraphConfig):
        self.cfg = cfg
        p = gtsam.ISAM2Params()
        p.setRelinearizeThreshold(cfg.relinearize_threshold)
        p.relinearizeSkip = 1
        if cfg.isam_optimizer == "dogleg":
            # Trust region: plain Gauss-Newton can diverge numerically when
            # many conflicting (e.g. false) links pull against odometry.
            p.setOptimizationParams(gtsam.ISAM2DoglegParams())
        self.isam = gtsam.ISAM2(p)
        self.base = gtsam.NonlinearFactorGraph()      # prior + odometry (never removed)
        self.initial = gtsam.Values()                 # full copy of initial guesses
        self._pending = gtsam.NonlinearFactorGraph()
        self._pending_tags: list[int | None] = []     # group id per pending factor (None = base)
        self._pending_values = gtsam.Values()
        self._estimate = gtsam.Values()   # full estimate, built lazily (see .estimate)
        self._estimate_valid = True
        self._in_isam: set[int] = set()     # node indices already inside iSAM2
        self.n = 0
        self.groups: dict[int, LinkGroup] = {}
        self.events: list[tuple[int, int, str]] = []  # (node, group, event)
        self._last_audit = 0
        self._loop_sigma = np.array(cfg.loop_sigma, dtype=float)
        self._loop_noise = self._make_loop_noise()

    # ------------------------------------------------------------------ build
    def _make_loop_noise(self):
        base = gtsam.noiseModel.Diagonal.Sigmas(self._loop_sigma)
        k = self.cfg.robust_k
        kind = self.cfg.robust_kernel.lower()
        if kind == "none":
            return base
        m = {"huber": gtsam.noiseModel.mEstimator.Huber,
             "cauchy": gtsam.noiseModel.mEstimator.Cauchy,
             "dcs": gtsam.noiseModel.mEstimator.DCS,
             "gm": gtsam.noiseModel.mEstimator.GemanMcClure}[kind].Create(k)
        return gtsam.noiseModel.Robust.Create(m, base)

    def add_first(self, pose=(0.0, 0.0, 0.0)) -> int:
        noise = gtsam.noiseModel.Diagonal.Sigmas(np.array(self.cfg.prior_sigma, dtype=float))
        self._add_base(gtsam.PriorFactorPose2(X(0), pose2(pose), noise))
        self._pending_values.insert(X(0), pose2(pose))
        self.initial.insert(X(0), pose2(pose))
        self.n = 1
        return 0

    def add_odometry(self, delta) -> int:
        """Append node n with odometry delta from node n-1. Returns its key index."""
        i = self.n
        d = float(np.hypot(delta[0], delta[1]))
        sig = np.array(self.cfg.odom_sigma_per_sqrt_m) * np.sqrt(d) + np.array(self.cfg.odom_sigma_floor)
        noise = gtsam.noiseModel.Diagonal.Sigmas(sig)
        self._add_base(gtsam.BetweenFactorPose2(X(i - 1), X(i), pose2(delta), noise))
        prev = self.pose(i - 1) if (i - 1) in self._in_isam else self._pending_values.atPose2(X(i - 1))
        guess = prev.compose(pose2(delta))
        self._pending_values.insert(X(i), guess)
        self.initial.insert(X(i), guess)
        self.n += 1
        return i

    def add_view_link(self, query_key: int, anchor_key: int, measured, hypothesis_id: int = -1,
                      similarity: float = float("nan")) -> None:
        """Weak link (anchor pose in the query frame), filed under its hypothesis' group."""
        g = self.groups.get(hypothesis_id)
        if g is None:
            g = self.groups[hypothesis_id] = LinkGroup(hypothesis_id, created_key=query_key)
            g.history.append((query_key, "committed"))
        f = gtsam.BetweenFactorPose2(X(query_key), X(anchor_key), pose2(measured), self._loop_noise)
        link = LoopLink(query_key, anchor_key, tuple(float(m) for m in measured), hypothesis_id,
                        similarity, "active" if g.active else "rejected")
        g.links.append(link)
        g.factors.append(f)
        if g.active:                         # a rejected group keeps collecting, but off-graph
            self._pending.add(f)
            self._pending_tags.append(g.id)

    def _add_base(self, f) -> None:
        self._pending.add(f)
        self._pending_tags.append(None)
        self.base.add(f)

    # --------------------------------------------------------------- optimise
    def update(self, extra_iterations: int = 0, remove: list[int] | None = None) -> None:
        if remove:
            r = self.isam.update(self._pending, self._pending_values, remove)
        else:
            r = self.isam.update(self._pending, self._pending_values)
        for idx, tag in zip(r.getNewFactorsIndices(), self._pending_tags):
            if tag is not None:
                g = self.groups[tag]
                g.isam_idx = (g.isam_idx or []) + [int(idx)]
        for _ in range(extra_iterations):
            self.isam.update()
        for k in self._pending_values.keys():
            self._in_isam.add(gtsam.Symbol(k).index())
        self._pending = gtsam.NonlinearFactorGraph()
        self._pending_tags = []
        self._pending_values = gtsam.Values()
        # Per-pulse cost: do NOT rebuild all poses here (O(n) every pulse);
        # single poses come from calculateEstimatePose2 on demand.
        self._estimate_valid = False
        self._marginals = None

    @property
    def estimate(self) -> gtsam.Values:
        """Full current estimate (all nodes) - built on demand, cached until the next update."""
        if not self._estimate_valid:
            self._estimate = self.isam.calculateEstimate()
            self._estimate_valid = True
        return self._estimate

    def _set_status(self, g: LinkGroup, status: str, key: int, event: str) -> None:
        g.status = status
        for l in g.links:
            l.status = "rejected" if status == "rejected" else "active"
        g.history.append((key, event))
        self.events.append((key, g.id, event))

    def remove_group(self, gid: int, key: int, event: str = "removed") -> None:
        g = self.groups[gid]
        idx = g.isam_idx or []
        g.isam_idx = None
        self._set_status(g, "rejected", key, event)
        # drop still-pending factors of this group as well
        if any(t == gid for t in self._pending_tags):
            keep = gtsam.NonlinearFactorGraph()
            tags = []
            for i, t in enumerate(self._pending_tags):
                if t != gid:
                    keep.add(self._pending.at(i))
                    tags.append(t)
            self._pending, self._pending_tags = keep, tags
        self.update(extra_iterations=self.cfg.extra_iterations_on_loop, remove=idx)

    def readmit_group(self, gid: int, key: int) -> None:
        g = self.groups[gid]
        self._set_status(g, "tentative", key, "readmitted")
        for f in g.factors:
            self._pending.add(f)
            self._pending_tags.append(g.id)
        self.update(extra_iterations=self.cfg.extra_iterations_on_loop)

    # ------------------------------------------------------- link management
    def link_chi2(self, link: LoopLink, values=None) -> float:
        """Plain (non-robust) Mahalanobis^2 of one link at the given estimate."""
        v = self.estimate if values is None else values
        pred = v.atPose2(X(link.query_key)).between(v.atPose2(X(link.anchor_key)))
        e = gtsam.Pose2.Logmap(pose2(link.measured).between(pred)) / self._loop_sigma
        return float(e @ e)

    def group_score(self, g: LinkGroup, values=None) -> float:
        return float(np.median([self.link_chi2(l, values) for l in g.links])) if g.links else 0.0

    def manage(self, key: int, force_audit: bool = False, new_links: bool = True) -> list[int]:
        """Length rule + fast consistency check + periodic GNC audit. Returns ids
        of groups that are (newly) rejected, so the recogniser can drop them."""
        if not self.cfg.link_management or not self.groups:
            return []
        removed: list[int] = []
        # length rule: withdraw groups that ended short (brief aliases)
        if self.cfg.min_group_links > 0:
            for g in list(self.groups.values()):
                if (g.active and len(g.links) < self.cfg.min_group_links
                        and g.links[-1].query_key < key - self.cfg.short_group_grace):
                    g.permanent_reject = True
                    self.remove_group(g.id, key, f"withdrawn: only {len(g.links)} links")
                    removed.append(g.id)
        if not new_links and not force_audit and not (
                self.cfg.gnc_every > 0 and key - self._last_audit >= self.cfg.gnc_every):
            return removed
        # fast check: throw out the worst inconsistent TENTATIVE group, repeat
        for _ in range(5):
            cand = [(self.group_score(g), g.id) for g in self.groups.values() if g.status == "tentative"]
            if not cand:
                break
            worst, gid = max(cand)
            if worst <= self.cfg.group_chi2_max:
                break
            self.remove_group(gid, key, f"removed: residual chi2 {worst:.1f}")
            removed.append(gid)
        # periodic GNC audit over everything (active AND rejected groups)
        due = self.cfg.gnc_every > 0 and key - self._last_audit >= self.cfg.gnc_every
        if due or force_audit:
            self._last_audit = key
            removed += self.audit(key)
        return removed

    def audit(self, key: int) -> list[int]:
        groups = [g for g in self.groups.values() if g.links and not g.permanent_reject]
        if not groups:
            return []
        graph = gtsam.NonlinearFactorGraph()
        for i in range(self.base.size()):
            graph.add(self.base.at(i))
        n_base = graph.size()
        spans = []
        plain = gtsam.noiseModel.Diagonal.Sigmas(self._loop_sigma)
        for g in groups:
            start = graph.size()
            for l in g.links:     # GNC does its own robustification: give it plain Gaussians
                graph.add(gtsam.BetweenFactorPose2(X(l.query_key), X(l.anchor_key), pose2(l.measured), plain))
            spans.append((g, start, graph.size()))
        params = gtsam.GncLMParams()
        params.setLossType(gtsam.GncLossType.GM)
        params.setKnownInliers(list(range(n_base)))
        opt = gtsam.GncLMOptimizer(graph, self.estimate, params)
        opt.optimize()
        w = np.asarray(opt.getWeights())
        removed = []
        for g, a, b in spans:
            mw = float(w[a:b].mean())
            keep = mw >= self.cfg.gnc_min_weight
            if keep and not g.active:
                self.readmit_group(g.id, key)
                g.history.append((key, f"audit weight {mw:.2f}"))
            elif not keep and g.active:
                self.remove_group(g.id, key, f"removed: GNC weight {mw:.2f}")
                removed.append(g.id)
            elif keep:
                g.audits_passed += 1
                if g.status == "tentative" and g.audits_passed >= 2:
                    self._set_status(g, "confirmed", key, f"confirmed (GNC weight {mw:.2f})")
        return removed

    # ------------------------------------------------------------------ query
    @property
    def loops(self) -> list[LoopLink]:
        """Links currently in the graph."""
        return [l for g in self.groups.values() if g.active for l in g.links]

    @property
    def all_links(self) -> list[LoopLink]:
        return [l for g in self.groups.values() for l in g.links]

    def active_graph(self) -> gtsam.NonlinearFactorGraph:
        graph = gtsam.NonlinearFactorGraph()
        for i in range(self.base.size()):
            graph.add(self.base.at(i))
        for g in self.groups.values():
            if g.active:
                for f in g.factors:
                    graph.add(f)
        return graph

    def batch_optimize(self) -> np.ndarray:
        """Levenberg-Marquardt on the active graph from the current estimate."""
        init = self.estimate if self.estimate.size() == self.n else self.initial
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(100)
        res = gtsam.LevenbergMarquardtOptimizer(self.active_graph(), init, params).optimize()
        return np.array([[res.atPose2(X(i)).x(), res.atPose2(X(i)).y(), res.atPose2(X(i)).theta()]
                         for i in range(self.n)])

    def pose(self, i: int) -> gtsam.Pose2:
        if self._estimate_valid:
            return self._estimate.atPose2(X(i))
        return self.isam.calculateEstimatePose2(X(i))

    def poses(self) -> np.ndarray:
        v = self.estimate
        return np.array([[v.atPose2(X(i)).x(), v.atPose2(X(i)).y(), v.atPose2(X(i)).theta()] for i in range(self.n)])

    def marginal_covariance(self, i: int) -> np.ndarray:
        return self.isam.marginalCovariance(X(i))

    def relative_covariance(self, a: int, b: int) -> np.ndarray:
        """Covariance of the relative pose x_a^-1 x_b (in the frame of b), from the joint
        marginal of both nodes. Much tighter than the marginal of b alone when a and b share
        their drift from the origin, which is what a loop closure a~b actually tests."""
        if getattr(self, "_marginals", None) is None:    # one factorisation per graph state
            self._marginals = gtsam.Marginals(self.active_graph(), self.estimate)
        S = self._marginals.jointMarginalCovariance(gtsam.KeyVector([X(a), X(b)])).fullMatrix()
        Saa, Sbb, Sab = S[:3, :3], S[3:, 3:], S[:3, 3:]
        rel = self.pose(a).between(self.pose(b))
        Ha = -rel.inverse().AdjointMap()       # d(rel)/d(x_a); d(rel)/d(x_b) = I
        return Ha @ Saa @ Ha.T + Sbb + Ha @ Sab + Sab.T @ Ha.T
