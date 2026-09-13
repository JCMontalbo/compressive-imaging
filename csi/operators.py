"""Linear measurement operators as forward / adjoint pairs.

Every solver in :mod:`csi.solvers` only needs ``A(x)``, ``A.H(y)`` and a bound on ``‖A‖²``; nothing
here forms a dense matrix unless the operator *is* one. Complex-valued throughout where the
measurement is Fourier.
"""

from __future__ import annotations

import numpy as np


class LinearOperator:
    """Base: subclasses implement ``_forward`` and ``_adjoint``; ``shape_in`` is the signal shape."""

    shape_in: tuple
    shape_out: tuple

    def __call__(self, x):
        return self._forward(np.asarray(x))

    def H(self, y):
        return self._adjoint(np.asarray(y))

    def norm_sq(self, iters: int = 30, seed: int = 0) -> float:
        """Upper estimate of ‖A‖² by power iteration (the Lipschitz constant of ½‖Ax−y‖² is ‖A‖²)."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.shape_in) + 1j * rng.standard_normal(self.shape_in)
        x /= np.linalg.norm(x)
        s = 0.0
        for _ in range(iters):
            z = self.H(self(x))
            s = float(np.linalg.norm(z))
            x = z / max(s, 1e-30)
        return s * 1.05

    def adjoint_check(self, seed: int = 0) -> float:
        """|<Ax, y> − <x, Aᴴy>| / (‖Ax‖‖y‖): ~1e-12 when the adjoint is right."""
        rng = np.random.default_rng(seed)
        x = rng.standard_normal(self.shape_in) + 1j * rng.standard_normal(self.shape_in)
        y = rng.standard_normal(self.shape_out) + 1j * rng.standard_normal(self.shape_out)
        ax, ahy = self(x), self.H(y)
        lhs = np.vdot(y, ax)
        rhs = np.vdot(ahy, x)
        return float(abs(lhs - rhs) / (np.linalg.norm(ax) * np.linalg.norm(y) + 1e-30))


class Dense(LinearOperator):
    """An explicit matrix ``A`` (m × n)."""

    def __init__(self, A: np.ndarray):
        self.A = np.asarray(A)
        self.shape_in = (self.A.shape[1],)
        self.shape_out = (self.A.shape[0],)

    def _forward(self, x):
        return self.A @ x

    def _adjoint(self, y):
        return self.A.conj().T @ y


def gaussian_matrix(m: int, n: int, seed: int = 0) -> Dense:
    """i.i.d. N(0, 1/m) sensing matrix — the classic CS measurement."""
    rng = np.random.default_rng(seed)
    return Dense(rng.standard_normal((m, n)) / np.sqrt(m))


class PartialFourier(LinearOperator):
    """Keep the rows ``idx`` of the unitary DFT of a signal (1-D) or image (2-D).

    ``y = F_Ω x`` with ``F`` the orthonormal (i)FFT, so the adjoint is the zero-filled inverse — the
    "backprojection" baseline. Works on any number of dimensions via ``np.fft.fftn``.
    """

    def __init__(self, shape_in, idx):
        self.shape_in = tuple(shape_in)
        self.idx = np.asarray(idx)  # flat indices into the fftn output
        self.shape_out = (len(self.idx),)

    def _forward(self, x):
        return np.fft.fftn(x, norm="ortho").ravel()[self.idx]

    def _adjoint(self, y):
        z = np.zeros(int(np.prod(self.shape_in)), complex)
        z[self.idx] = y
        return np.fft.ifftn(z.reshape(self.shape_in), norm="ortho")

    def zero_fill(self, y):
        """Alias for the adjoint: the conventional reconstruction from missing Fourier data."""
        return self._adjoint(y)


def random_fourier_mask(shape, fraction: float, seed: int = 0, keep_dc: bool = True):
    """Flat indices of a uniformly random ``fraction`` of Fourier coefficients."""
    rng = np.random.default_rng(seed)
    n = int(np.prod(shape))
    m = int(round(fraction * n))
    idx = rng.choice(n, size=m, replace=False)
    if keep_dc and 0 not in idx:
        idx[0] = 0
    return np.sort(idx)


def radial_lines_mask(n: int, n_lines: int):
    """Flat indices of ``n_lines`` radial lines through the centre of an n×n Fourier plane
    (the Candès–Romberg–Tao phantom experiment; also the thesis §6.2)."""
    mask = np.zeros((n, n), bool)
    c = n // 2
    for k in range(n_lines):
        th = np.pi * k / n_lines
        for r in np.arange(-c, c, 0.5):
            i = int(round(c + r * np.cos(th)))
            j = int(round(c + r * np.sin(th)))
            if 0 <= i < n and 0 <= j < n:
                mask[i, j] = True
    return np.flatnonzero(np.fft.ifftshift(mask).ravel())


class NonUniformSamples(LinearOperator):
    """Time-domain samples ``x[t_idx]`` of a signal that is sparse in the DFT basis.

    The unknown is the DFT coefficient vector ``c``; ``x = F⁻¹c``; measurements are the kept
    time samples. This is the thesis's ch. 6 setting (a "permuted DFT dictionary").
    """

    def __init__(self, n: int, t_idx):
        self.n = n
        self.t_idx = np.asarray(t_idx)
        self.shape_in = (n,)
        self.shape_out = (len(self.t_idx),)

    def _forward(self, c):
        return np.fft.ifft(c, norm="ortho")[self.t_idx]

    def _adjoint(self, y):
        x = np.zeros(self.n, complex)
        x[self.t_idx] = y
        return np.fft.fft(x, norm="ortho")


class Compose(LinearOperator):
    """``A ∘ B``: measure with ``A`` a signal synthesised by ``B`` (e.g. Gaussian matrix ∘ inverse DCT)."""

    def __init__(self, A: LinearOperator, B: LinearOperator):
        self.A, self.B = A, B
        self.shape_in, self.shape_out = B.shape_in, A.shape_out

    def _forward(self, x):
        return self.A(self.B(x))

    def _adjoint(self, y):
        return self.B.H(self.A.H(y))
