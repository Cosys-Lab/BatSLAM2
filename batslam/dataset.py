"""Load a dataset written by ``sim/generate_dataset.m``."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.io


@dataclass
class Reflectors:
    position: np.ndarray        # [M x 3] nominal position
    type: list[str]             # "point" | "circle" | "wall"
    radius: np.ndarray          # [M]
    endpoint1: np.ndarray       # [M x 3] (walls)
    endpoint2: np.ndarray       # [M x 3] (walls)
    has_signature: np.ndarray   # [M] bool


@dataclass
class Dataset:
    gt_pose: np.ndarray         # [N x 3] x, y, theta (rad)
    gt_delta: np.ndarray        # [N-1 x 3] body-frame increments
    odom_delta: np.ndarray      # [N-1 x 3] noisy body-frame increments
    t: np.ndarray               # [N]
    lap: np.ndarray             # [N] route segment index (1-based)
    lap_names: list[str]
    echo_left_i16: np.ndarray   # [N x L] int16
    echo_right_i16: np.ndarray
    echo_scale: np.ndarray      # [N]
    fs: float
    speed_of_sound: float
    call: np.ndarray            # emitted call waveform
    max_range: float
    reflectors: Reflectors
    meta: dict

    @property
    def n(self) -> int:
        return self.gt_pose.shape[0]

    def echo(self, i: int) -> tuple[np.ndarray, np.ndarray]:
        """Binaural echo of pulse i as float64 (left, right)."""
        s = self.echo_scale[i]
        return (self.echo_left_i16[i].astype(np.float64) * s,
                self.echo_right_i16[i].astype(np.float64) * s)



def resynthesize_odometry(ds: Dataset, bias_factor: float = 1.0, noise_factor: float = 1.0,
                          seed: int = 1) -> Dataset:
    """Replace ds.odom_delta with fresh odometry from the ground-truth increments,
    using the dataset's own noise model (generate_dataset.m) with the systematic
    part (heading bias, scale error) scaled by ``bias_factor`` and the random
    part by ``noise_factor``. For drift-robustness sweeps without re-rendering."""
    m = ds.meta
    sig = np.asarray(m["odom_sigma"], float) * noise_factor
    bias_rot = float(m["odom_bias_rot"]) * bias_factor
    bias_scale = float(m["odom_bias_scale"]) * bias_factor
    rng = np.random.default_rng(seed)
    d = np.hypot(ds.gt_delta[:, 0], ds.gt_delta[:, 1])
    ds.odom_delta = (ds.gt_delta * np.array([1 + bias_scale, 1, 1])
                     + np.c_[np.zeros_like(d), np.zeros_like(d), bias_rot * d]
                     + sig * np.sqrt(d)[:, None] * rng.standard_normal(ds.gt_delta.shape))
    return ds


def load_dataset(path: str | Path) -> Dataset:
    m = scipy.io.loadmat(str(path), squeeze_me=True, struct_as_record=False)
    meta = m["meta"]
    r = m["reflectors"]
    types = [str(t).strip() for t in np.atleast_1d(r.type)]
    lap_names = [str(t).strip() for t in np.atleast_1d(meta.lap_names)]
    meta_d = {k: getattr(meta, k) for k in meta._fieldnames}
    return Dataset(
        gt_pose=np.atleast_2d(m["gt_pose"]).astype(float),
        gt_delta=np.atleast_2d(m["gt_delta"]).astype(float),
        odom_delta=np.atleast_2d(m["odom_delta"]).astype(float),
        t=np.atleast_1d(m["t"]).astype(float),
        lap=np.atleast_1d(m["lap"]).astype(int),
        lap_names=lap_names,
        echo_left_i16=np.atleast_2d(m["echo_left"]),
        echo_right_i16=np.atleast_2d(m["echo_right"]),
        echo_scale=np.atleast_1d(m["echo_scale"]).astype(float),
        fs=float(meta.fs),
        speed_of_sound=float(meta.speed_of_sound),
        call=np.asarray(meta.call, dtype=float).ravel(),
        max_range=float(meta.max_range),
        reflectors=Reflectors(
            position=np.atleast_2d(r.position),
            type=types,
            radius=np.atleast_1d(r.radius),
            endpoint1=np.atleast_2d(r.endpoint1),
            endpoint2=np.atleast_2d(r.endpoint2),
            has_signature=np.atleast_1d(r.has_signature).astype(bool),
        ),
        meta=meta_d,
    )
