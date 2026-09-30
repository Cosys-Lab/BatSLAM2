"""All BatSLAM 2.0 tunables in one place, grouped per pipeline stage.

Every stage takes only its own sub-config, so stages stay testable in
isolation; ``BatSLAMConfig`` just bundles them for the pipeline.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path


@dataclass
class FrontendConfig:
    """Echo -> local-view descriptor (a binaural 'cochleogram')."""
    matched_filter: bool = True        # dechirp with the emitted call first
    n_channels: int = 64               # fine band-pass filterbank, log-spaced
    band_pool: int = 2                 # average adjacent fine bands -> n_channels / band_pool bands
    f_low: float = 30e3                # Hz
    f_high: float = 90e3               # Hz
    range_bin: float = 0.06            # m per descriptor range bin
    min_range: float = 0.25            # m, blank near field
    max_range: float | None = None     # m; None = the dataset's own max range
    range_blur_bins: float = 1.0       # Gaussian blur along range, in bins (1 bin = 6 cm: pose tolerance)
    range_gain_exponent: float = 1.0   # time-varying gain: amplitude x r^k (k=2 undoes spherical spreading)
    # descriptor: "energy_shape" = compressed energy image + spectral-shape (direction) image,
    # each normalised separately; "energy" = the energy image alone (legacy)
    descriptor: str = "energy_shape"
    compression: str = "power"         # "power" (amplitude^power_exponent) | "log"
    power_exponent: float = 0.33       # cube-root compression, relative to the view peak
    log_floor_db: float = -30.0        # only for compression="log": dB floor below the view peak
    shape_floor_db: float = -40.0      # spectral shape is only kept where echo power is above this
    shape_scale_db: float = 15.0       # shape values are clipped to +- this (in dB) and scaled to +-1


@dataclass
class TemplateConfig:
    """Pose-anchored local-view templates."""
    spacing_m: float = 0.45            # min travel (odometry) between new templates
    spacing_rad: float = 0.35          # ...or this much heading change
    max_shift_bins: int = 3            # range-shift search in similarity (+-bins; 3 x 6 cm = +-18 cm)
    whiten: str = "none"               # "none" | "zscore" (running per-feature standardisation)
    whiten_warmup: int = 50            # views before whitening kicks in
    whiten_refresh: int = 100          # re-whiten the template matrix every n views
    # local view = stack of the last history_n views, history_spacing_m apart
    # (odometric distance); 1 = single-view local views
    history_n: int = 1
    history_spacing_m: float = 0.3
    top_k: int = 5                     # candidates returned per query
    min_similarity: float = 0.60       # similarity floor for a candidate (P(right) < 1 % below)
    exclude_recent_m: float = 6.0      # ignore templates laid within this much recent travel
    # Optional pose gating on the current graph estimate (None = global search).
    gate_radius_m: float | None = None
    gate_heading_rad: float | None = 1.2  # views are heading-dependent: +-~70 deg


@dataclass
class RecognitionConfig:
    """Sequence-based, multi-hypothesis verification of recognitions."""
    # similarity -> log-likelihood ratio (match vs non-match), logistic-ish linear
    # set once from the empirical evidence curve of the development drive (indoor_r3):
    # P(right | s) = 0.5 near s = 0.70; +2 / +4 at s = 0.75 / 0.80
    llr_slope: float = 40.0
    llr_mid: float = 0.70              # similarity at which LLR = 0
    llr_clip: float = 4.0
    miss_penalty: float = 0.5          # evidence lost per step without support
    max_gap: int = 4                   # steps a hypothesis survives without support
    # geometric consistency between consecutive pairs of a hypothesis
    tol_xy_m: float = 0.30
    tol_xy_frac: float = 0.15          # + fraction of distance travelled since last pair
    tol_rad: float = 0.25
    # commit rule
    min_pairs: int = 8
    min_templates: int = 3
    commit_evidence: float = 10.0
    ambiguity_margin: float = 4.0      # best must beat best *inconsistent* rival by this
    plausibility_chi2: float | None = 16.27   # 3-dof 99.9% gate on the implied correction
    # for corrections > defer_correction_m, gate on the anchor->now relative covariance
    # (joint marginal) instead of the current node's marginal
    plausibility_relative: bool = True
    plausibility_relative_min_m: float = 1.5   # (same value as defer_correction_m, but independent)
    # risk-scaled commit: a hypothesis whose commit would move the map by more
    # than defer_correction_m must first be long AND strong: at least
    # defer_min_pairs consistent pairs, a total evidence of defer_min_evidence
    # (four times the normal commit_evidence) and a mean evidence per pair of
    # defer_min_llr_per_pair. Brief aliases die before, long aliases along a
    # quasi-regular structure (a pillar lane) match only weakly (< 1.15 per pair
    # for all wrong large-correction commits in the main experiments), while true revisits
    # match strongly (median 2.8 per pair) and commit after about 2.4 m.
    defer_correction_m: float | None = 1.5
    defer_min_pairs: int = 16
    defer_min_evidence: float = 40.0
    defer_min_llr_per_pair: float = 1.0
    max_hypotheses: int = 200


@dataclass
class GraphConfig:
    """GTSAM pose graph."""
    prior_sigma: tuple[float, float, float] = (0.01, 0.01, 0.005)
    # odometry noise the back-end ASSUMES (per sqrt(m) travelled, plus floor)
    odom_sigma_per_sqrt_m: tuple[float, float, float] = (0.04, 0.02, 0.035)
    odom_sigma_floor: tuple[float, float, float] = (0.005, 0.005, 0.003)
    # weak view-recognition link
    loop_sigma: tuple[float, float, float] = (0.30, 0.30, 0.20)
    robust_kernel: str = "huber"       # "huber" | "cauchy" | "dcs" | "gm" | "none"
    robust_k: float = 1.345
    use_shift_offset: bool = True      # turn the range shift into a forward offset
    relinearize_threshold: float = 0.01
    isam_optimizer: str = "dogleg"     # "dogleg" (trust region) | "gauss-newton"
    # multi-hypothesis link management (see backend/pose_graph.py)
    link_management: bool = True
    group_chi2_max: float = 12.0       # fast check: median per-link chi2 (3 dof) of a tentative group
    gnc_every: int = 400               # nodes between GNC audits (0 = never)
    gnc_min_weight: float = 0.5        # group mean GNC weight needed to stay in the graph
    min_group_links: int = 16          # a group ending with fewer links is withdrawn for good (0 = off)
    short_group_grace: int = 10        # pulses without a new link after which a group has 'ended'
    extra_iterations_on_loop: int = 3


@dataclass
class BatSLAMConfig:
    frontend: FrontendConfig = field(default_factory=FrontendConfig)
    templates: TemplateConfig = field(default_factory=TemplateConfig)
    recognition: RecognitionConfig = field(default_factory=RecognitionConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)

    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "BatSLAMConfig":
        return _update(cls(), d)

    @classmethod
    def load(cls, path: str | Path) -> "BatSLAMConfig":
        return cls.from_dict(json.loads(Path(path).read_text()))


def _update(obj, d: dict):
    """Recursively apply a (partial) dict onto a dataclass instance."""
    names = {f.name for f in fields(obj)}
    for k, v in d.items():
        if k not in names:
            raise KeyError(f"unknown config key {k!r} for {type(obj).__name__}")
        cur = getattr(obj, k)
        if is_dataclass(cur) and isinstance(v, dict):
            _update(cur, v)
        else:
            setattr(obj, k, tuple(v) if isinstance(cur, tuple) else v)
    return obj
