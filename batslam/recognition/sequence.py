"""Sequence-based verification of local-view recognitions.

Single sonar views are ambiguous (corridors look alike, and the hall has
a deliberately point-symmetric aliasing trap). A single recognition is
therefore never trusted. Instead every candidate recognition is either

  * appended to an existing *hypothesis* it is geometrically consistent
    with, or
  * the seed of a new hypothesis.

A hypothesis is a chain of (query node, template) pairs claiming "the
stretch I am driving now is the stretch I drove back then". It is
consistent if the motion between consecutive queries (odometry - locally
accurate) matches the motion between their templates' anchors (current
map estimate - also locally accurate), within a tolerance that grows
with distance travelled.

Each hypothesis accumulates evidence = sum of log-likelihood ratios of
its similarities, and bleeds evidence on steps without support. Several
mutually exclusive hypotheses can live at once - that *is* the ambiguity
- and one is committed only when it is

  1. long enough (pairs and distinct templates),
  2. strong enough (evidence threshold), and
  3. unambiguous: it beats the best *inconsistent* rival (a hypothesis
     that puts the robot somewhere else) by a margin, and
  4. risk-scaled: if committing would move the map by more than
     ``defer_correction_m``, the hypothesis must first reach
     ``defer_min_pairs`` consistent pairs with a mean evidence of at least
     ``defer_min_llr_per_pair`` per pair (a wrong commit's damage grows with
     the correction it forces; aliases are brief, or long but weak);
  5. plausible: the pose correction it implies fits the current pose
     uncertainty (``plausible`` callback, a Mahalanobis gate on the
     back-end's covariance). This is what rejects *globally*
     symmetric aliases (e.g. a corridor that looks like the opposite
     corridor rotated by 180 deg): a whole sequence can be internally
     consistent and still imply an impossible jump. For large corrections
     the gate uses the relative covariance between anchor and current node.

Committed hypotheses release all their pairs as weak links for the pose
graph, and every further consistent pair is released immediately while
the hypothesis stays alive ("locked on").
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import gtsam
import numpy as np

from ..config import RecognitionConfig
from ..views.templates import Candidate

PoseFn = Callable[[int], gtsam.Pose2]


@dataclass
class Hypothesis:
    id: int
    pairs: list[Candidate]
    evidence: float
    gap: int = 0
    committed: bool = False
    n_released: int = 0
    extended_now: bool = True
    born_key: int = 0

    @property
    def last(self) -> Candidate:
        return self.pairs[-1]

    @property
    def n_templates(self) -> int:
        return len({p.template_id for p in self.pairs})


@dataclass
class StepResult:
    release: list[tuple[Candidate, int]] = field(default_factory=list)  # (pair, hypothesis id)
    locked: bool = False                    # a committed hypothesis was extended this step
    n_alive: int = 0
    best_evidence: float = 0.0
    rival_evidence: float = 0.0
    committed_ids: list[int] = field(default_factory=list)


class SequenceVerifier:
    def __init__(self, cfg: RecognitionConfig, odom_pose: PoseFn, map_pose: PoseFn,
                 use_shift_offset: bool = True,
                 plausible: Callable[[gtsam.Pose2, int, int], tuple] | None = None):
        self.cfg = cfg
        self.odom_pose = odom_pose      # dead-reckoned pose of a query node
        self.map_pose = map_pose        # current graph estimate of an anchor node
        self.use_shift = use_shift_offset
        self.plausible = plausible      # (implied current pose, now key, anchor key) -> (d2, dist, dtheta)
        self.hyps: list[Hypothesis] = []
        self._next_id = 0
        self.history: list[StepResult] = []
        self.n_rejected_implausible = 0
        self.commit_log: list[dict] = []   # one record per commit (for analysis / tuning)

    # ------------------------------------------------------------- geometry
    def measurement(self, c: Candidate) -> gtsam.Pose2:
        """Anchor pose in the query frame implied by one recognition."""
        return gtsam.Pose2(c.shift_m if self.use_shift else 0.0, 0.0, 0.0)

    def llr(self, sim: float) -> float:
        v = self.cfg.llr_slope * (sim - self.cfg.llr_mid)
        return float(np.clip(v, -self.cfg.llr_clip, self.cfg.llr_clip))

    def consistent(self, a: Candidate, b: Candidate) -> bool:
        pa = self.odom_pose(a.query_key).compose(self.measurement(a))
        pb = self.odom_pose(b.query_key).compose(self.measurement(b))
        dq = pa.between(pb)
        dt = self.map_pose(a.anchor_key).between(self.map_pose(b.anchor_key))
        e = dq.between(dt)
        travelled = float(np.hypot(dq.x(), dq.y()))
        tol = self.cfg.tol_xy_m + self.cfg.tol_xy_frac * travelled
        return float(np.hypot(e.x(), e.y())) < tol and abs(e.theta()) < self.cfg.tol_rad

    def implied_pose(self, h: Hypothesis, now_key: int) -> gtsam.Pose2:
        """Where hypothesis h says the robot is *now*, in the map frame."""
        c = h.last
        q_in_map = self.map_pose(c.anchor_key).compose(self.measurement(c).inverse())
        drift = q_in_map.compose(self.odom_pose(c.query_key).inverse())   # odom -> map
        return drift.compose(self.odom_pose(now_key))

    def _agree(self, h1: Hypothesis, h2: Hypothesis, now_key: int) -> bool:
        e = self.implied_pose(h1, now_key).between(self.implied_pose(h2, now_key))
        return np.hypot(e.x(), e.y()) < 1.0 and abs(e.theta()) < 0.4

    def drop(self, hypothesis_id: int) -> None:
        """The back-end rejected this hypothesis' links: forget it."""
        self.hyps = [h for h in self.hyps if h.id != hypothesis_id]

    def is_locked(self) -> bool:
        return any(h.committed and h.extended_now for h in self.hyps)

    # ----------------------------------------------------------------- step
    def step(self, query_key: int, candidates: list[Candidate]) -> StepResult:
        cfg = self.cfg
        res = StepResult()
        used = set()

        # 1. extend existing hypotheses with their best consistent candidate
        for h in self.hyps:
            h.extended_now = False
            best, best_llr = None, -np.inf
            for ci, c in enumerate(candidates):
                # short baseline (last pair) AND long baseline (first pair)
                if self.consistent(h.last, c) and (len(h.pairs) < 3 or self.consistent(h.pairs[0], c)):
                    l = self.llr(c.similarity)
                    if l > best_llr:
                        best, best_llr = ci, l
            if best is not None:
                h.pairs.append(candidates[best])
                h.evidence += best_llr
                h.gap = 0
                h.extended_now = True
                used.add(best)
            else:
                h.gap += 1
                h.evidence -= cfg.miss_penalty

        # 2. unexplained candidates seed new hypotheses
        for ci, c in enumerate(candidates):
            if ci in used:
                continue
            self.hyps.append(Hypothesis(self._next_id, [c], self.llr(c.similarity), born_key=query_key))
            self._next_id += 1

        # 3. prune: dead, weak, or duplicate (same last template -> keep strongest)
        alive = [h for h in self.hyps
                 if h.gap <= cfg.max_gap and (h.committed or h.evidence > -2.0)]
        alive.sort(key=lambda h: (h.committed, h.evidence), reverse=True)
        seen, dedup = set(), []
        for h in alive:
            key = (h.last.template_id, h.last.query_key)
            if key in seen and not h.committed:
                continue
            seen.add(key)
            dedup.append(h)
        self.hyps = dedup[: cfg.max_hypotheses]

        # 4. commit decision among uncommitted, eligible hypotheses
        eligible = [h for h in self.hyps if not h.committed and h.extended_now
                    and len(h.pairs) >= cfg.min_pairs and h.n_templates >= cfg.min_templates
                    and h.evidence >= cfg.commit_evidence]
        if eligible and self.plausible is not None:
            for h in eligible:
                h.plaus = self.plausible(self.implied_pose(h, query_key), query_key, h.last.anchor_key)
            lim = cfg.plausibility_chi2
            rejected = [h for h in eligible if lim is not None and h.plaus[0] > lim]
            if cfg.defer_correction_m is not None:     # large correction: wait for more pairs
                eligible = [h for h in eligible if h in rejected or h.plaus[1] <= cfg.defer_correction_m
                            or (len(h.pairs) >= cfg.defer_min_pairs
                                and h.evidence / len(h.pairs) >= cfg.defer_min_llr_per_pair
                                and h.evidence >= cfg.defer_min_evidence)]
            for h in rejected:          # implausible jump: kill, and remember why
                h.gap = cfg.max_gap + 1
                self.n_rejected_implausible += 1
            eligible = [h for h in eligible if h not in rejected]
            self.hyps = [h for h in self.hyps if h not in rejected]
        if eligible:
            best = max(eligible, key=lambda h: h.evidence)
            rivals = [h.evidence for h in self.hyps
                      if h is not best and not self._agree(h, best, query_key)]
            rival = max(rivals, default=0.0)
            res.best_evidence, res.rival_evidence = best.evidence, rival
            if best.evidence - rival >= cfg.ambiguity_margin:
                best.committed = True
                res.committed_ids.append(best.id)
                sims = np.array([p.similarity for p in best.pairs])
                a, b = self.odom_pose(best.pairs[0].query_key), self.odom_pose(best.pairs[-1].query_key)
                d2, dxy, dth = getattr(best, "plaus", (float("nan"),) * 3)
                self.commit_log.append(dict(
                    id=best.id, key=query_key, n_pairs=len(best.pairs), n_templates=best.n_templates,
                    evidence=best.evidence, rival=rival, n_alive=len(self.hyps),
                    sim_mean=float(sims.mean()), sim_min=float(sims.min()),
                    span_m=float(np.hypot(b.x() - a.x(), b.y() - a.y())),
                    plaus_d2=d2, corr_m=dxy, corr_rad=dth))
                # hypotheses agreeing with the winner are redundant now
                self.hyps = [h for h in self.hyps
                             if h is best or h.committed or not self._agree(h, best, query_key)]

        # 5. release links from committed hypotheses
        for h in self.hyps:
            if h.committed and h.n_released < len(h.pairs):
                res.release.extend((p, h.id) for p in h.pairs[h.n_released:])
                h.n_released = len(h.pairs)
            if h.committed and h.extended_now:
                res.locked = True

        res.n_alive = len(self.hyps)
        if not eligible and self.hyps:
            res.best_evidence = max(h.evidence for h in self.hyps)
        self.history.append(res)
        return res
