"""Sparsifying transforms as operators: DCT, Walsh–Hadamard, finite differences (for TV).

Each is a :class:`csi.operators.LinearOperator` whose *forward* map goes from coefficients to signal
(synthesis), so it composes with a measurement operator as ``Compose(measure, synthesis)``.
"""

from __future__ import annotations

import numpy as np
from scipy.fft import dct, idct

from .operators import LinearOperator


class InverseDCT(LinearOperator):
    """Orthonormal DCT-II synthesis: signal = idct(coefficients). 1-D or separable 2-D."""

    def __init__(self, shape):
        self.shape_in = self.shape_out = tuple(shape)

    def _forward(self, c):
        out = c
        for ax in range(c.ndim):
            out = idct(out, norm="ortho", axis=ax)
        return out

    def _adjoint(self, x):
        out = x
        for ax in range(x.ndim):
            out = dct(out, norm="ortho", axis=ax)
        return out


def fwht(x: np.ndarray) -> np.ndarray:
    """Fast Walsh–Hadamard transform (Sylvester ordering), orthonormal. Length must be a power of two."""
    x = np.array(x, dtype=complex if np.iscomplexobj(x) else float)
    n = x.shape[-1]
    assert n & (n - 1) == 0, "length must be a power of two"
    h = 1
    while h < n:
        x = x.reshape(-1, n // (2 * h), 2, h)
        a, b = x[:, :, 0, :].copy(), x[:, :, 1, :].copy()
        x[:, :, 0, :], x[:, :, 1, :] = a + b, a - b
        x = x.reshape(-1, n)
        h *= 2
    return x.reshape(-1) / np.sqrt(n) if x.shape[0] == 1 else x / np.sqrt(n)


class InverseWHT(LinearOperator):
    """Walsh–Hadamard synthesis; the transform is its own inverse (orthonormal, symmetric)."""

    def __init__(self, n: int):
        self.shape_in = self.shape_out = (n,)

    def _forward(self, c):
        return fwht(c)

    def _adjoint(self, x):
        return fwht(x)


def grad2d(x: np.ndarray):
    """Forward differences with zero at the far boundary; returns (dx, dy)."""
    dx = np.zeros_like(x)
    dy = np.zeros_like(x)
    dx[:, :-1] = x[:, 1:] - x[:, :-1]
    dy[:-1, :] = x[1:, :] - x[:-1, :]
    return dx, dy


def div2d(px: np.ndarray, py: np.ndarray):
    """Negative adjoint of :func:`grad2d`: ``div = -gradᵀ``."""
    dx = np.zeros_like(px)
    dy = np.zeros_like(py)
    dx[:, 0] = px[:, 0]
    dx[:, 1:-1] = px[:, 1:-1] - px[:, :-2]
    dx[:, -1] = -px[:, -2]
    dy[0, :] = py[0, :]
    dy[1:-1, :] = py[1:-1, :] - py[:-2, :]
    dy[-1, :] = -py[-2, :]
    return dx + dy


def tv(x: np.ndarray) -> float:
    dx, dy = grad2d(x)
    return float(np.sqrt(np.abs(dx) ** 2 + np.abs(dy) ** 2).sum())
