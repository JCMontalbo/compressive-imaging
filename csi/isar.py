"""Turntable ISAR: the model of Hu, Montalbo, Li, Sun & Qiao (SPIE 9857, 2016), built properly.

Far-field, Born-approximation, stepped-frequency phase history of a set of point scatterers on a
turntable rotating through a small angle:

    D(f_m, θ_n) = Σ_i σ_i exp(−j 4π f_m r_i(θ_n) / c),   r_i(θ) = x_i cos θ + y_i sin θ.

With  k_x = 2f cosθ / c,  k_y = 2f sinθ / c  this is the 2-D Fourier transform of the reflectivity
map sampled on a polar grid (range–Doppler / polar format). Imaging = interpolate the polar samples
onto a rectangular k-grid and inverse-FFT. The paper's parameters (Table 1): 3.0–3.384 GHz,
20 kHz PRF, 0.15 rad/s, 1300 m. Compressive sampling: keep a random subset of the phase-history samples.

Two reconstruction pipelines on the same kept samples (docs/plan.md, Part 3):

* ``pipeline_paper``  -- the paper's Sec. 3: treat the *data* as the sparse unknown, ℓ₁-recover it
                         from partial Fourier rows, then image the recovered data.
* ``pipeline_image``  -- treat the kept samples as partial Fourier measurements of the *image* and
                         ℓ₁-recover the image directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.interpolate import griddata

from .operators import PartialFourier
from .solvers import basis_pursuit_admm, fista

C = 299_792_458.0


@dataclass
class Radar:
    f0: float = 3.0e9  # start frequency (Hz)
    bandwidth: float = 384e6
    n_freq: int = 64  # stepped-frequency samples per pulse
    n_pulse: int = 64  # pulses over the coherent processing interval
    omega: float = 0.15  # turntable rate (rad/s)
    prf: float = 20e3

    @property
    def freqs(self):
        return self.f0 + np.arange(self.n_freq) * self.bandwidth / self.n_freq

    @property
    def dtheta(self):
        """Total rotation over the CPI. The paper's 20 kHz pulses at 0.15 rad/s give a tiny angle per
        pulse; we keep ``n_pulse`` pulses spread over the aperture needed for the chosen cross-range
        resolution (decimated slow time), so that range and cross-range cells are the same size."""
        lam = C / (self.f0 + self.bandwidth / 2)
        return lam / (2 * self.range_res)

    @property
    def thetas(self):
        return (np.arange(self.n_pulse) - self.n_pulse / 2) * self.dtheta / self.n_pulse

    @property
    def range_res(self):
        return C / (2 * self.bandwidth)

    @property
    def cpi_seconds(self):
        return self.dtheta / self.omega


@dataclass
class Scene:
    x: np.ndarray  # metres, range direction
    y: np.ndarray  # metres, cross-range
    amp: np.ndarray

    @property
    def n(self):
        return len(self.x)


def pseudo_aircraft(spacing: float = 1.0) -> Scene:
    """The paper's Figure 2: fuselage along x (−10..10), wing at y = −5 (−5..5), tail at x = 0 (0..5)."""
    pts = [(x, 0.0) for x in np.arange(-10, 10.01, spacing)]
    pts += [(x, -5.0) for x in np.arange(-5, 5.01, spacing)]
    pts += [(0.0, y) for y in np.arange(1.0, 5.01, spacing)]
    xs, ys = zip(*pts)
    return Scene(np.array(xs), np.array(ys), np.ones(len(pts)))


def phase_history(scene: Scene, radar: Radar) -> np.ndarray:
    """Exact far-field phase history, shape [n_pulse, n_freq]."""
    f = radar.freqs[None, None, :]
    th = radar.thetas[None, :, None]
    r = scene.x[:, None, None] * np.cos(th) + scene.y[:, None, None] * np.sin(th)
    return np.sum(scene.amp[:, None, None] * np.exp(-1j * 4 * np.pi * f * r / C), axis=0)


def polar_k(radar: Radar):
    """k-space coordinates of each phase-history sample: kx = 2f cosθ/c, ky = 2f sinθ/c."""
    f = radar.freqs[None, :]
    th = radar.thetas[:, None]
    return 2 * f * np.cos(th) / C, 2 * f * np.sin(th) / C


def image_grid(radar: Radar, n: int):
    """An n×n image with pixel = resolution cell, centred on the turntable axis."""
    d = radar.range_res
    ax = (np.arange(n) - n // 2) * d
    return ax, ax


def polar_to_rect(D: np.ndarray, radar: Radar, n: int) -> np.ndarray:
    """Interpolate the polar phase history onto the rectangular k-grid that the n×n image's FFT lives on
    (the paper's Figure 1). Returns the rectangular data, fftshifted so that index 0 is DC."""
    kx, ky = polar_k(radar)
    d = radar.range_res
    dk = 1 / (n * d)  # k-grid spacing for an n-pixel image with pixel d
    kc = 2 * (radar.f0 + radar.bandwidth / 2) / C
    kxg = kc + (np.arange(n) - n // 2) * dk
    kyg = (np.arange(n) - n // 2) * dk
    KX, KY = np.meshgrid(kxg, kyg, indexing="xy")
    pts = np.column_stack([kx.ravel(), ky.ravel()])
    re = griddata(pts, D.real.ravel(), (KX, KY), method="linear", fill_value=0.0)
    im = griddata(pts, D.imag.ravel(), (KX, KY), method="linear", fill_value=0.0)
    rect = re + 1j * im
    # remove the carrier: the image is the FT w.r.t. (kx − kc, ky)
    return np.fft.ifftshift(rect)


def image_from_rect(rect: np.ndarray) -> np.ndarray:
    """Inverse 2-D FFT of rectangular-format data -> complex image (fftshifted to centre)."""
    return np.fft.fftshift(np.fft.ifft2(rect, norm="ortho"))


def render_scene(scene: Scene, radar: Radar, n: int) -> np.ndarray:
    """The scatterers rendered on the pixel grid (nearest cell): the sparse image the CS model targets."""
    d = radar.range_res
    img = np.zeros((n, n), complex)
    for x, y, a in zip(scene.x, scene.y, scene.amp):
        i = int(round(y / d)) + n // 2
        j = int(round(x / d)) + n // 2
        if 0 <= i < n and 0 <= j < n:
            img[i, j] += a
    return img


def rect_data_from_image(img: np.ndarray) -> np.ndarray:
    return np.fft.fft2(np.fft.ifftshift(img), norm="ortho")


# --------------------------------------------------------------------------- compressive pipelines


def keep_random(n_samples: int, fraction: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_samples, int(round(fraction * n_samples)), replace=False))


def pipeline_zero_fill(rect: np.ndarray, idx) -> np.ndarray:
    """Conventional: put the kept samples back, zeros elsewhere, inverse FFT."""
    z = np.zeros(rect.size, complex)
    z[idx] = rect.ravel()[idx]
    return image_from_rect(z.reshape(rect.shape))


def pipeline_paper(rect: np.ndarray, idx, n_iter: int = 400) -> np.ndarray:
    """The paper's Sec. 3: the *data* d is the unknown, measured through random rows of the DFT,
    recovered as the sparsest z with F_Ω z = y, then imaged. (The data is not sparse; see H3.)"""
    A = PartialFourier(rect.shape, idx)
    y = A(rect)
    z = basis_pursuit_admm(A, y, n_iter=n_iter).x
    return image_from_rect(z)


def pipeline_image(rect: np.ndarray, idx, lam: float | None = None, n_iter: int = 300) -> np.ndarray:
    """The kept samples are partial Fourier measurements of the image: y = P F x. ℓ₁-recover x."""
    y = rect.ravel()[idx]
    A = _ImageMeasure(rect.shape, idx)
    lam = lam if lam is not None else 0.02 * np.abs(A.H(y)).max()
    return fista(A, y, lam, n_iter=n_iter).x


class _ImageMeasure(PartialFourier):
    """P·F on a *centred* image (fftshift conventions matching image_from_rect)."""

    def _forward(self, x):
        return np.fft.fft2(np.fft.ifftshift(x), norm="ortho").ravel()[self.idx]

    def _adjoint(self, y):
        z = np.zeros(int(np.prod(self.shape_in)), complex)
        z[self.idx] = y
        return np.fft.fftshift(np.fft.ifft2(z.reshape(self.shape_in), norm="ortho"))


# --------------------------------------------------------------------------- scoring


def detect_peaks(img: np.ndarray, n_expected: int, min_rel: float = 0.3):
    """Local maxima of |img| above ``min_rel`` of the global max, strongest first, up to 2·n_expected."""
    a = np.abs(img)
    from scipy.ndimage import maximum_filter

    peaks = (a == maximum_filter(a, size=3)) & (a >= min_rel * a.max())
    ii, jj = np.nonzero(peaks)
    order = np.argsort(-a[ii, jj])[: 2 * n_expected]
    return np.column_stack([ii[order], jj[order]])


def f1_score(img: np.ndarray, truth: np.ndarray, tol_cells: float = 1.0):
    """Match detected peaks to true scatterer cells within ``tol_cells``; return (F1, mean localisation error)."""
    ti, tj = np.nonzero(np.abs(truth) > 0)
    tpts = np.column_stack([ti, tj]).astype(float)
    det = detect_peaks(img, len(tpts)).astype(float)
    if len(det) == 0:
        return 0.0, np.inf
    d = np.linalg.norm(det[:, None, :] - tpts[None, :, :], axis=-1)
    matched_t, matched_d, errs = set(), set(), []
    for k in np.argsort(d, axis=None):
        i, j = np.unravel_index(k, d.shape)
        if d[i, j] > tol_cells:
            break
        if i in matched_d or j in matched_t:
            continue
        matched_d.add(i)
        matched_t.add(j)
        errs.append(d[i, j])
    tp = len(matched_t)
    prec = tp / len(det)
    rec = tp / len(tpts)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return float(f1), float(np.mean(errs)) if errs else np.inf


def sparsity_energy(v: np.ndarray, k: int) -> float:
    """Fraction of ‖v‖² captured by its k largest entries (in the basis v is given in)."""
    a = np.sort(np.abs(v.ravel()))[::-1]
    return float((a[:k] ** 2).sum() / (a**2).sum())
