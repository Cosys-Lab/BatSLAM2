"""Pose-anchored local-view templates.

A template is a descriptor *plus the pose-graph node it was recorded at*.
Its pose (x, y, theta) is therefore never stored as a number: it is
whatever the pose graph currently believes about the anchor node, so
templates move with every optimisation. Recognising a template is
evidence that the robot is (nearly) at that anchor pose - including the
heading, since sonar views are strongly heading dependent.

Similarity is a range-shift-tolerant normalised correlation: the query is
compared at a few range offsets and the best one is kept. Optionally
(``whiten="zscore"``) every feature is first standardised with running
statistics over all views seen so far: in self-similar places (streets)
most of a view is the same strong wall echo everywhere, and plain
correlation then mostly measures "I am in a street". Whitening makes what
is *unusual* about a view dominate the match. The winning
shift is itself informative - a query whose echoes arrive later than the
template's is standing *behind* the anchor - and is turned into a
forward-offset estimate for the loop factor.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import TemplateConfig


@dataclass
class Template:
    id: int
    anchor_key: int             # pose-graph node index
    descriptor: np.ndarray      # [2, C, B]
    odom_s: float               # odometric arc length at creation


@dataclass
class Candidate:
    """One single-view recognition: query node ~ template anchor."""
    query_key: int
    template_id: int
    anchor_key: int
    similarity: float
    shift_m: float              # + = query is behind the anchor (echoes later)


def _normalise(v: np.ndarray) -> np.ndarray:
    v = v - v.mean(axis=-1, keepdims=True)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, 1e-12)


class TemplateStore:
    def __init__(self, cfg: TemplateConfig, descriptor_shape: tuple[int, int, int], range_bin: float):
        self.cfg = cfg
        self.shape = descriptor_shape
        self.range_bin = range_bin
        self.templates: list[Template] = []
        D = int(np.prod(descriptor_shape))
        # growable row buffers (amortised O(1) add; a vstack per add is O(N^2)):
        # raw descriptors, and their whitened + normalised form used for matching
        self._raw = np.zeros((256, D), dtype=np.float32)
        self._buf = np.zeros((256, D), dtype=np.float32)
        # running feature statistics (Welford) and the frozen copy in use
        self._n_obs = 0
        self._mean = np.zeros(D)
        self._m2 = np.zeros(D)
        self._mu = np.zeros(D, dtype=np.float32)
        self._sd = np.ones(D, dtype=np.float32)
        self._since_refresh = 0

    @property
    def _mat(self) -> np.ndarray:
        return self._buf[: len(self.templates)]

    # ------------------------------------------------------------ whitening
    def observe(self, descriptor: np.ndarray) -> None:
        """Feed every view (not only templates) into the running statistics."""
        if self.cfg.whiten == "none":
            return
        x = descriptor.reshape(-1).astype(np.float64)
        self._n_obs += 1
        d = x - self._mean
        self._mean += d / self._n_obs
        self._m2 += d * (x - self._mean)
        self._since_refresh += 1
        if self._n_obs >= self.cfg.whiten_warmup and (
                self._since_refresh >= self.cfg.whiten_refresh or self._n_obs == self.cfg.whiten_warmup):
            self._refresh()

    def _refresh(self) -> None:
        self._mu = self._mean.astype(np.float32)
        self._sd = (np.sqrt(self._m2 / max(self._n_obs - 1, 1)) + 1e-3).astype(np.float32)
        n = len(self.templates)
        if n:
            self._buf[:n] = _normalise(self._whiten(self._raw[:n]))
        self._since_refresh = 0

    def _whiten(self, X: np.ndarray) -> np.ndarray:
        if self.cfg.whiten == "none" or self._n_obs < self.cfg.whiten_warmup:
            return X
        return (X - self._mu) / self._sd

    def __len__(self) -> int:
        return len(self.templates)

    def add(self, anchor_key: int, descriptor: np.ndarray, odom_s: float) -> Template:
        t = Template(len(self.templates), anchor_key, descriptor.copy(), odom_s)
        n = len(self.templates)
        if n == self._buf.shape[0]:
            self._buf = np.concatenate([self._buf, np.zeros_like(self._buf)])
            self._raw = np.concatenate([self._raw, np.zeros_like(self._raw)])
        self._raw[n] = descriptor.reshape(-1)
        self._buf[n] = _normalise(self._whiten(self._raw[n : n + 1]))[0]
        self.templates.append(t)
        return t

    def _shifted_queries(self, descriptor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S = self.cfg.max_shift_bins
        shifts = np.arange(-S, S + 1)
        descriptor = self._whiten(descriptor.reshape(1, -1).astype(np.float32)).reshape(descriptor.shape)
        B = descriptor.shape[-1]
        Q = np.zeros((len(shifts),) + descriptor.shape, dtype=np.float32)
        for i, s in enumerate(shifts):
            # Q_s[..., b] = descriptor[..., b + s]: s > 0 pulls later echoes
            # forward, i.e. a positive best shift means the query sees
            # things further away than the template did.
            if s >= 0:
                Q[i, ..., : B - s] = descriptor[..., s:]
            else:
                Q[i, ..., -s:] = descriptor[..., : B + s]
        return _normalise(Q.reshape(len(shifts), -1)), shifts

    def similarities(self, descriptor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Best similarity and best shift (bins) against every template."""
        if not self.templates:
            return np.zeros(0), np.zeros(0, dtype=int)
        Q, shifts = self._shifted_queries(descriptor)
        sims = Q @ self._mat.T                       # [n_shift, n_templates]
        best = sims.argmax(axis=0)
        return sims[best, np.arange(sims.shape[1])], shifts[best]

    def query(self, query_key: int, descriptor: np.ndarray, odom_s: float,
              admissible=None) -> list[Candidate]:
        """Top-k candidates. ``admissible(template) -> bool`` adds pose gating."""
        sim, shift = self.similarities(descriptor)
        if sim.size == 0:
            return []
        order = np.argsort(-sim)
        out: list[Candidate] = []
        for j in order:
            if sim[j] < self.cfg.min_similarity or len(out) >= self.cfg.top_k:
                break
            t = self.templates[j]
            if odom_s - t.odom_s < self.cfg.exclude_recent_m:
                continue
            if admissible is not None and not admissible(t):
                continue
            out.append(Candidate(query_key, t.id, t.anchor_key, float(sim[j]),
                                 float(shift[j]) * self.range_bin))
        return out
