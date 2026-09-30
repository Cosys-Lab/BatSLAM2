"""Binaural echo -> local-view descriptor.

The descriptor is a *cochleogram*: per ear, per frequency channel, the
log echo envelope as a function of range. This is the representation the
original BatSLAM used for its local views - it keeps the full acoustic
"fingerprint" of a place (who echoes, how strong, at what range, in
which band, in which ear) without trying to extract landmarks.

Implementation: one FFT per ear. In the frequency domain we matched-filter
with the emitted call and apply one-sided Gaussian band masks on a fine
log-spaced filterbank (64 bands) - a one-sided spectrum inverse-transforms
to the analytic signal, so ``abs(ifft(...))`` is the band envelope. Power
is averaged into range bins and adjacent bands are pooled.

The default descriptor ("energy_shape") has two images, each normalised
separately so that neither dominates the correlation:
  * energy: time-varying gain (amplitude x r), then compression relative to
    the view peak (cube root by default);
  * spectral shape: per ear and range bin, the dB spectrum minus its mean
    over bands, where there is echo energy. This is the direction cue: the
    ear and emitter directivity (HRTF) shapes the spectrum of each echo by
    direction, and removing the level makes it comparable across range.
Both are blurred slightly along range so that a template tolerates a few
centimetres of displacement.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d

from ..config import FrontendConfig


class Cochleogram:
    def __init__(self, cfg: FrontendConfig, fs: float, speed_of_sound: float,
                 call: np.ndarray, n_samples: int, max_range: float = 5.0):
        self.cfg = cfg
        self.fs = fs
        self.c = speed_of_sound
        self.n_samples = n_samples
        self.nfft = int(2 ** np.ceil(np.log2(n_samples + len(call))))
        freqs = np.fft.fftfreq(self.nfft, 1.0 / fs)

        # Band masks: log-spaced centre frequencies, Gaussian on log-f,
        # neighbouring channels crossing at ~half power. One-sided (x2).
        fc = np.geomspace(cfg.f_low, cfg.f_high, cfg.n_channels)
        self.center_freqs = fc
        log_step = np.log(fc[1] / fc[0]) if cfg.n_channels > 1 else 0.5
        sigma = log_step / 2.355 * 2.0
        pos = freqs > 0
        H = np.zeros((cfg.n_channels, self.nfft))
        lf = np.log(np.where(pos, freqs, 1.0))
        for i, f in enumerate(fc):
            H[i, pos] = 2.0 * np.exp(-0.5 * ((lf[pos] - np.log(f)) / sigma) ** 2)
        if cfg.matched_filter:
            C = np.fft.fft(call, self.nfft)
            H = H * np.conj(C)[None, :] / (np.abs(C).max() + 1e-12)
        self.H = H

        # Range bins: sample index k <-> range k / fs * c / 2
        self.max_range = cfg.max_range if cfg.max_range is not None else max_range
        edges_m = np.arange(cfg.min_range, self.max_range + 1e-9, cfg.range_bin)
        self.range_centers = 0.5 * (edges_m[:-1] + edges_m[1:])
        edges_idx = np.round(edges_m * 2.0 / self.c * fs).astype(int)
        self._edges = np.clip(edges_idx, 0, n_samples)

        # Fast path: each band is non-zero only on a slice of spectrum bins
        # [k0, k0 + span). The inverse FFT of that slice at length L >= span
        # equals the full-rate analytic band signal sampled every D = nfft/L
        # samples (times L/nfft and a pure phase term), so its magnitude is
        # the exact envelope at the decimated instants - the decimated rate
        # (fs/D) stays above twice the band bandwidth by construction (so that
        # the band power |y|^2 is sampled without aliasing).
        spans = []
        for i in range(H.shape[0]):
            nz = np.nonzero(np.abs(H[i]) > 1e-6 * np.abs(H[i]).max())[0]
            spans.append((int(nz[0]), int(nz[-1]) + 1))
        # x2: the POWER |y|^2 that is bin-averaged has twice the band's bandwidth
        L = 2 * int(2 ** np.ceil(np.log2(max(b - a for a, b in spans))))
        self._slices = spans
        self._L = min(L, self.nfft)
        self._D = self.nfft // self._L
        self._Hs = np.zeros((H.shape[0], self._L), dtype=np.complex128)
        for i, (a, b) in enumerate(spans):
            self._Hs[i, : b - a] = H[i, a:b]
        self._edges_dec = np.clip(np.round(self._edges / self._D).astype(int), 0, (n_samples + self._D - 1) // self._D)

    @property
    def n_bands(self) -> int:
        return self.cfg.n_channels // max(self.cfg.band_pool, 1)

    @property
    def shape(self) -> tuple[int, int, int]:
        blocks = 2 if self.cfg.descriptor == "energy_shape" else 1
        return (2 * blocks, self.n_bands, len(self.range_centers))

    def _ear(self, x: np.ndarray) -> np.ndarray:
        X = np.fft.fft(x, self.nfft)
        if self._D > 1:                                    # decimated per-band inverse FFT (exact samples)
            Z = np.zeros_like(self._Hs)
            for i, (a, b) in enumerate(self._slices):
                Z[i, : b - a] = X[a:b]
            env = np.abs(np.fft.ifft(self._Hs * Z, axis=1)) * (self._L / self.nfft)
            env = env[:, : (self.n_samples + self._D - 1) // self._D]
            cs = np.concatenate([np.zeros((env.shape[0], 1)), np.cumsum(env ** 2, axis=1)], axis=1)
            a, b = self._edges_dec[:-1], self._edges_dec[1:]
            return (cs[:, b] - cs[:, a]) / np.maximum(b - a, 1)
        env = np.abs(np.fft.ifft(self.H * X[None, :], axis=1))[:, : self.n_samples]
        # bin-average via cumulative sums
        cs = np.concatenate([np.zeros((env.shape[0], 1)), np.cumsum(env ** 2, axis=1)], axis=1)
        a, b = self._edges[:-1], self._edges[1:]
        power = (cs[:, b] - cs[:, a]) / np.maximum(b - a, 1)
        return power

    def power(self, left: np.ndarray, right: np.ndarray, pooled: bool = True) -> np.ndarray:
        """Linear envelope power per ear/band/range bin: [2, n_bands, n_range]."""
        p = np.stack([self._ear(left), self._ear(right)])
        k = max(self.cfg.band_pool, 1)
        if pooled and k > 1:
            p = p[:, : self.n_bands * k].reshape(2, self.n_bands, k, -1).mean(axis=2)
        return p

    def _blur(self, v: np.ndarray) -> np.ndarray:
        if self.cfg.range_blur_bins > 0:
            v = gaussian_filter1d(v, self.cfg.range_blur_bins, axis=-1, mode="constant")
        return v

    @staticmethod
    def _centred_unit(v: np.ndarray) -> np.ndarray:
        v = v - v.mean()
        return v / max(float(np.linalg.norm(v)), 1e-12)

    def energy(self, rel: np.ndarray) -> np.ndarray:
        """Compressed energy image from peak-relative power."""
        if self.cfg.compression == "power":
            return rel ** (self.cfg.power_exponent / 2)                   # amplitude^gamma
        floor = self.cfg.log_floor_db
        db = 10.0 * np.log10(rel + 1e-30)
        return (np.clip(db, floor, 0.0) - floor) / (-floor)

    def spectral_shape(self, rel: np.ndarray) -> np.ndarray:
        """Per ear and range bin: dB spectrum minus its mean over bands (masked, scaled to +-1)."""
        db = 10.0 * np.log10(rel + 1e-30)
        mask = db > self.cfg.shape_floor_db
        s = (db - db.mean(axis=1, keepdims=True)) / self.cfg.shape_scale_db
        return np.clip(s, -1.0, 1.0) * mask

    def compute(self, left: np.ndarray, right: np.ndarray) -> np.ndarray:
        """Return the descriptor for one pulse: [2 (x2 with shape), n_bands, n_range]."""
        p = self.power(left, right)
        if self.cfg.range_gain_exponent:
            p = p * self.range_centers[None, None, :] ** (2 * self.cfg.range_gain_exponent)
        rel = p / (p.max() + 1e-30)
        e = self._blur(self.energy(rel))
        if self.cfg.descriptor != "energy_shape":
            return e.astype(np.float32)
        s = self._blur(self.spectral_shape(rel))
        return np.concatenate([self._centred_unit(e), self._centred_unit(s)], axis=0).astype(np.float32)


def compute_all(dataset, cfg: FrontendConfig, progress: bool = True) -> tuple[np.ndarray, Cochleogram]:
    """Descriptors for every pulse of a dataset: [N, 2, C, B]."""
    fe = Cochleogram(cfg, dataset.fs, dataset.speed_of_sound, dataset.call,
                     dataset.echo_left_i16.shape[1], dataset.max_range)
    out = np.zeros((dataset.n,) + fe.shape, dtype=np.float32)
    for i in range(dataset.n):
        out[i] = fe.compute(*dataset.echo(i))
        if progress and (i + 1) % 200 == 0:
            print(f"  frontend {i + 1}/{dataset.n}")
    return out, fe
