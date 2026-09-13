"""The master's thesis's signals, sampler and FTSA, as written (Montalbo 2016, ch. 4–6)."""

from __future__ import annotations

import numpy as np

N = 400
T = np.linspace(-1, 1, N)  # the thesis's time axis (both endpoints included: not periodic)


def sine_train(r, amplitudes=None) -> np.ndarray:
    """§4.1:  f(t) = Σ aᵢ sin(cᵢ t),  aᵢ = 10ⁱ,  cᵢ = aᵢ · rᵢ^{rᵢ},  rᵢ ∈ {1..10}.

    With 400 samples on [−1, 1] these frequencies (up to ~10¹⁴ rad/s) are aliased many times over;
    the discrete signal is five tones at aliased, generally off-bin frequencies.
    """
    r = np.asarray(r, float)
    a = 10.0 ** np.arange(1, len(r) + 1) if amplitudes is None else np.asarray(amplitudes, float)
    c = a * r**r
    return sum(ai * np.sin(ci * T) for ai, ci in zip(a, c))


def chirp_train(r, f0: float = 0.0) -> np.ndarray:
    """§4.5: sum of linear chirps from f0 to 10·rᵢ over the window (φ₀ = 0)."""
    out = np.zeros(N)
    for ri in r:
        f1 = 10.0 * ri
        k = (f1 - f0) / 2.0  # T = 2
        out += np.sin(2 * np.pi * (f0 * T + 0.5 * k * T**2))
    return out


def on_bin_tones(bins, amplitudes) -> np.ndarray:
    """Tones that are exactly periodic on the 400-sample grid: exactly 2·len(bins)-sparse in the DFT."""
    k = np.arange(N)
    return sum(a * np.sin(2 * np.pi * b * k / N) for a, b in zip(amplitudes, bins))


def nus_indices(rng, n: int = N, max_step: int = 10) -> np.ndarray:
    """The thesis's non-uniform sampler: start at randint(1, 10), advance by randint(1, 10), stop past n."""
    idx = []
    e = int(rng.integers(1, max_step + 1))
    while e <= n:
        idx.append(e - 1)
        e += int(rng.integers(1, max_step + 1))
    return np.array(idx)


def ftsa(f: np.ndarray, K: float):
    """Fourier Thresholding Selection Algorithm (§4.1): keep |F(ω)| ≥ max|F|/K, inverse-transform.

    Needs the whole signal — it is transform coding, not compressive sensing (thesis p. 41).
    Returns (reconstruction, number of coefficients kept).
    """
    F = np.fft.fft(f)
    keep = np.abs(F) >= np.abs(F).max() / K
    return np.fft.ifft(F * keep).real, int(keep.sum())


def wht_tsa(f: np.ndarray, K: float):
    """The same thresholding in the Walsh–Hadamard basis (§4.3); length is padded to a power of two."""
    from .transforms import fwht

    n = len(f)
    m = 1 << (n - 1).bit_length()
    g = np.zeros(m)
    g[:n] = f
    W = fwht(g)
    keep = np.abs(W) >= np.abs(W).max() / K
    return fwht(W * keep)[:n], int(keep.sum())


def cos_sin_dictionary(n: int, t_idx) -> np.ndarray:
    """Real dictionary of DC + cos/sin pairs on the n-grid, restricted to the sampled rows: the thesis's
    'permuted DFT' for a real signal, usable with the real interior-point solver."""
    k = np.arange(n)[:, None]
    m = n // 2
    freqs = np.arange(1, m)[None, :]
    D = np.hstack([np.ones((n, 1)), np.cos(2 * np.pi * k * freqs / n), np.sin(2 * np.pi * k * freqs / n)])
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    return D[t_idx]


def synth_from_dictionary(coef: np.ndarray, n: int) -> np.ndarray:
    """Inverse of :func:`cos_sin_dictionary` on the full grid."""
    k = np.arange(n)[:, None]
    m = n // 2
    freqs = np.arange(1, m)[None, :]
    D = np.hstack([np.ones((n, 1)), np.cos(2 * np.pi * k * freqs / n), np.sin(2 * np.pi * k * freqs / n)])
    D /= np.linalg.norm(D, axis=0, keepdims=True)
    return D @ coef


def shepp_logan(n: int) -> np.ndarray:
    """The Shepp–Logan phantom (standard ellipse parameters), n×n in [0, 1]."""
    ell = [  # (intensity, a, b, x0, y0, phi degrees)
        (1.0, 0.69, 0.92, 0, 0, 0), (-0.8, 0.6624, 0.874, 0, -0.0184, 0), (-0.2, 0.11, 0.31, 0.22, 0, -18),
        (-0.2, 0.16, 0.41, -0.22, 0, 18), (0.1, 0.21, 0.25, 0, 0.35, 0), (0.1, 0.046, 0.046, 0, 0.1, 0),
        (0.1, 0.046, 0.046, 0, -0.1, 0), (0.1, 0.046, 0.023, -0.08, -0.605, 0), (0.1, 0.023, 0.023, 0, -0.606, 0),
        (0.1, 0.023, 0.046, 0.06, -0.605, 0),
    ]
    y, x = np.mgrid[-1 : 1 : n * 1j, -1 : 1 : n * 1j]
    img = np.zeros((n, n))
    for a0, a, b, x0, y0, phi in ell:
        p = np.deg2rad(phi)
        xr = (x - x0) * np.cos(p) + (y - y0) * np.sin(p)
        yr = -(x - x0) * np.sin(p) + (y - y0) * np.cos(p)
        img[(xr / a) ** 2 + (yr / b) ** 2 <= 1] += a0
    return np.clip(img, 0, None) / img.max()
