"""Sparse-recovery solvers, written out rather than imported.

All take a :class:`csi.operators.LinearOperator` ``A`` (forward + adjoint) except the interior-point
method, which needs the dense matrix. ``soft`` is the complex soft-threshold, so the proximal solvers
work for Fourier measurements.

* :func:`ista`, :func:`fista`  -- proximal gradient for  min ½‖Ax−y‖² + λ‖x‖₁  (Daubechies 2004;
  Beck & Teboulle 2009 — the 2015 paper's "FGbCS" is FISTA with L the Lipschitz constant)
* :func:`omp`, :func:`cosamp`  -- greedy (Tropp & Gilbert 2007; Needell & Tropp 2009)
* :func:`irls`                 -- iteratively reweighted least squares for basis pursuit
* :func:`basis_pursuit_admm`   -- ADMM for  min ‖x‖₁ s.t. Ax = y  (Boyd et al. 2011 §6.2)
* :func:`basis_pursuit_ip`     -- primal log-barrier interior point for the same LP (what l1-magic's
                                  l1eq solved; Boyd & Vandenberghe ch. 11)
* :func:`tv_admm`              -- ADMM for  min ½‖Ax−y‖² + λ TV(x)  on images
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .operators import LinearOperator
from .transforms import div2d, grad2d


def soft(x, t):
    """Complex soft-threshold: shrink the modulus by ``t``, keep the phase."""
    mag = np.abs(x)
    return x * np.maximum(1 - t / np.maximum(mag, 1e-30), 0)


@dataclass
class Result:
    x: np.ndarray
    objective: list = field(default_factory=list)  # per iteration
    iters: int = 0
    residual: float = 0.0


# --------------------------------------------------------------------------- proximal gradient


def _lasso_objective(A, x, y, lam):
    r = A(x) - y
    return 0.5 * float(np.vdot(r, r).real) + lam * float(np.abs(x).sum())


def ista(A: LinearOperator, y, lam: float, n_iter: int = 500, L: float | None = None, x0=None, tol: float = 1e-8):
    L = L or A.norm_sq()
    x = np.zeros(A.shape_in, complex) if x0 is None else np.array(x0, complex)
    obj = []
    for k in range(n_iter):
        g = A.H(A(x) - y)
        x_new = soft(x - g / L, lam / L)
        obj.append(_lasso_objective(A, x_new, y, lam))
        if np.linalg.norm(x_new - x) <= tol * max(np.linalg.norm(x), 1e-30):
            x = x_new
            break
        x = x_new
    return Result(x, obj, k + 1, float(np.linalg.norm(A(x) - y)))


def fista(A: LinearOperator, y, lam: float, n_iter: int = 500, L: float | None = None, x0=None, tol: float = 1e-8):
    """Beck–Teboulle: momentum t_{k+1} = (1 + √(1 + 4t_k²))/2, O(1/k²)."""
    L = L or A.norm_sq()
    x = np.zeros(A.shape_in, complex) if x0 is None else np.array(x0, complex)
    z, t = x.copy(), 1.0
    obj = []
    for k in range(n_iter):
        g = A.H(A(z) - y)
        x_new = soft(z - g / L, lam / L)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        z = x_new + ((t - 1) / t_new) * (x_new - x)
        obj.append(_lasso_objective(A, x_new, y, lam))
        if np.linalg.norm(x_new - x) <= tol * max(np.linalg.norm(x), 1e-30):
            x = x_new
            break
        x, t = x_new, t_new
    return Result(x, obj, k + 1, float(np.linalg.norm(A(x) - y)))


# --------------------------------------------------------------------------- greedy


def _lstsq_on_support(A: LinearOperator, y, support):
    """Least squares restricted to ``support`` using the dense sub-matrix (built column by column)."""
    n = int(np.prod(A.shape_in))
    cols = []
    for j in support:
        e = np.zeros(n, complex)
        e[j] = 1.0
        cols.append(A(e.reshape(A.shape_in)).ravel())
    As = np.stack(cols, axis=1)
    coef, *_ = np.linalg.lstsq(As, np.asarray(y).ravel(), rcond=None)
    x = np.zeros(n, complex)
    x[list(support)] = coef
    return x.reshape(A.shape_in), As @ coef


def omp(A: LinearOperator, y, k: int, tol: float = 1e-10):
    """Pick the column most correlated with the residual, re-fit, repeat k times."""
    y = np.asarray(y, complex)
    r = y.copy()
    support: list[int] = []
    x = np.zeros(A.shape_in, complex)
    obj = []
    for _ in range(k):
        c = np.abs(A.H(r)).ravel()
        c[support] = -1
        support.append(int(np.argmax(c)))
        x, fit = _lstsq_on_support(A, y, support)
        r = y - fit.reshape(y.shape)
        obj.append(float(np.linalg.norm(r)))
        if obj[-1] <= tol * np.linalg.norm(y):
            break
    return Result(x, obj, len(support), obj[-1] if obj else float(np.linalg.norm(y)))


def cosamp(A: LinearOperator, y, k: int, n_iter: int = 50, tol: float = 1e-10):
    """Identify 2k candidates from the residual proxy, merge with the current support, least-squares, prune to k."""
    y = np.asarray(y, complex)
    n = int(np.prod(A.shape_in))
    x = np.zeros(n, complex)
    r = y.copy()
    obj = []
    for it in range(n_iter):
        proxy = np.abs(A.H(r)).ravel()
        omega = np.argpartition(proxy, -2 * k)[-2 * k :]
        support = np.union1d(omega, np.flatnonzero(x))
        b, _ = _lstsq_on_support(A, y, support)
        b = b.ravel()
        keep = np.argpartition(np.abs(b), -k)[-k:]
        x = np.zeros(n, complex)
        x[keep] = b[keep]
        r = y - A(x.reshape(A.shape_in)).reshape(y.shape)
        obj.append(float(np.linalg.norm(r)))
        if obj[-1] <= tol * np.linalg.norm(y) or (it > 0 and abs(obj[-1] - obj[-2]) < 1e-12):
            break
    return Result(x.reshape(A.shape_in), obj, it + 1, obj[-1])


# --------------------------------------------------------------------------- basis pursuit


def irls(A: LinearOperator, y, n_iter: int = 60, eps: float = 1.0, p: float = 1.0, cg_iter: int = 200):
    """min ‖x‖_p s.t. Ax = y by reweighting: x = W Aᴴ (A W Aᴴ)⁻¹ y with W = diag(|x|^{2-p} + ε).

    ε is shrunk toward zero as the iterates settle (Daubechies, DeVore, Fornasier, Güntürk 2010).
    """
    y = np.asarray(y, complex).ravel()
    n = int(np.prod(A.shape_in))
    w = np.ones(n)
    x = np.zeros(n, complex)
    obj = []
    for _ in range(n_iter):
        # solve (A W Aᴴ) u = y by conjugate gradients on the normal operator, then x = W Aᴴ u
        def M(u):
            return A(((w * A.H(u.reshape(A.shape_out)).ravel())).reshape(A.shape_in)).ravel()

        u = _cg(M, y, cg_iter)
        x_new = w * A.H(u.reshape(A.shape_out)).ravel()
        obj.append(float(np.abs(x_new).sum()))
        if np.linalg.norm(x_new - x) < 1e-8 * max(np.linalg.norm(x), 1e-30):
            x = x_new
            break
        x = x_new
        srt = np.sort(np.abs(x))[::-1]
        eps = min(eps, srt[min(len(srt) - 1, max(int(0.1 * n), 1))] / n) if len(srt) > 1 else eps
        w = np.abs(x) ** (2 - p) + eps
    return Result(x.reshape(A.shape_in), obj, len(obj), float(np.linalg.norm(A(x.reshape(A.shape_in)).ravel() - y)))


def _cg(M, b, n_iter, tol=1e-12):
    x = np.zeros_like(b)
    r = b - M(x)
    p = r.copy()
    rr = np.vdot(r, r).real
    for _ in range(n_iter):
        Mp = M(p)
        alpha = rr / max(np.vdot(p, Mp).real, 1e-300)
        x = x + alpha * p
        r = r - alpha * Mp
        rr_new = np.vdot(r, r).real
        if np.sqrt(rr_new) < tol * np.linalg.norm(b):
            break
        p = r + (rr_new / rr) * p
        rr = rr_new
    return x


def basis_pursuit_admm(A: LinearOperator, y, n_iter: int = 2000, rho: float = 1.0, tol: float = 1e-8, cg_iter: int = 100):
    """Boyd et al. §6.2: x ← proj onto {Ax = y} of (z − u); z ← soft(x + u, 1/ρ); u ← u + x − z."""
    y = np.asarray(y, complex).ravel()
    scale = max(float(np.linalg.norm(y)), 1e-30)  # the problem is scale-invariant; the iteration is not
    y = y / scale
    n = int(np.prod(A.shape_in))
    z = np.zeros(n, complex)
    u = np.zeros(n, complex)
    obj = []

    def AAH(v):
        return A(A.H(v.reshape(A.shape_out)).reshape(A.shape_in)).ravel()

    for k in range(n_iter):
        v = z - u
        # projection: x = v + Aᴴ (A Aᴴ)⁻¹ (y − A v)
        rhs = y - A(v.reshape(A.shape_in)).ravel()
        x = v + A.H(_cg(AAH, rhs, cg_iter).reshape(A.shape_out)).ravel()
        z_new = soft(x + u, 1.0 / rho)
        u = u + x - z_new
        obj.append(float(np.abs(z_new).sum()))
        prim = np.linalg.norm(x - z_new)
        dual = rho * np.linalg.norm(z_new - z)
        z = z_new
        if prim < tol * max(np.linalg.norm(x), 1) and dual < tol * max(np.linalg.norm(u), 1):
            break
    z = z * scale
    return Result(z.reshape(A.shape_in), [o * scale for o in obj], k + 1, float(np.linalg.norm(A(z.reshape(A.shape_in)).ravel() - y * scale)))


def basis_pursuit_ip(A: np.ndarray, y: np.ndarray, tol: float = 1e-8, mu: float = 10.0, max_newton: int = 50):
    """Primal log-barrier interior point for  min ‖x‖₁  s.t. Ax = y  (real-valued).

    LP form: min Σuᵢ  s.t.  −uᵢ ≤ xᵢ ≤ uᵢ,  Ax = y.  Barrier  φ = −Σ log(u−x) − Σ log(u+x).
    For each barrier weight t we take Newton steps on  t·Σu + φ  subject to Ax = y, solving the
    KKT system by elimination (Boyd & Vandenberghe §10.2), with backtracking line search; t grows by
    ``mu`` until the duality gap 2n/t is below ``tol``. Returns the same Result as the others plus the
    final gap.
    """
    A = np.asarray(A, float)
    y = np.asarray(y, float).ravel()
    m, n = A.shape
    # strictly feasible start: least-norm solution, u = |x| + margin
    x = np.linalg.lstsq(A, y, rcond=None)[0]
    u = np.abs(x) * 1.1 + 0.1 * (np.abs(x).max() + 1e-6)
    t = max(1.0, n / max(float(u.sum()), 1e-9))
    obj = []
    total_newton = 0
    while 2 * n / t > tol and total_newton < 400:
        for _ in range(max_newton):
            fu1, fu2 = u - x, u + x  # both > 0
            # gradient and Hessian of t*sum(u) + phi w.r.t. (x, u); blocks are diagonal:
            #   hxx = huu = 1/f1² + 1/f2²,  hxu = 1/f2² − 1/f1²
            gx = 1 / fu1 - 1 / fu2
            gu = t - 1 / fu1 - 1 / fu2
            f1s, f2s = fu1**2, fu2**2
            # eliminate u. Schur complement s = hxx − hxu²/huu = 4/(f1² + f2²)  (stable form);
            # hxu/huu = (f1² − f2²)/(f1² + f2²);  1/huu = f1² f2²/(f1² + f2²).
            sum_sq = f1s + f2s
            Sinv = 0.25 * sum_sq
            ratio = (f1s - f2s) / sum_sq
            w = gx - ratio * gu
            # solve [S Aᵀ; A 0][dx; nu] = [-w; 0]  ->  dx = -S⁻¹(w + Aᵀnu), A S⁻¹ Aᵀ nu = -A S⁻¹ w
            ASA = (A * Sinv) @ A.T
            nu = np.linalg.solve(ASA, -(A @ (Sinv * w)))
            dx = -Sinv * (w + A.T @ nu)
            hxu = 1 / f2s - 1 / f1s
            du = -(gu + hxu * dx) * (f1s * f2s / sum_sq)  # = -(gu + hxu dx)/huu
            total_newton += 1
            # Newton decrement
            lam2 = -(gx @ dx + gu @ du)
            if lam2 / 2 < 1e-10:
                break
            # backtracking keeping u ± x > 0
            step = 1.0
            while np.any(u + step * du - np.abs(x + step * dx) <= 0):
                step *= 0.5
            f0 = t * u.sum() - np.log(fu1).sum() - np.log(fu2).sum()
            while True:
                xn, un = x + step * dx, u + step * du
                fn = t * un.sum() - np.log(un - xn).sum() - np.log(un + xn).sum()
                if fn <= f0 - 0.01 * step * lam2 or step < 1e-10:
                    break
                step *= 0.5
            x, u = xn, un
        obj.append(float(np.abs(x).sum()))
        t *= mu
    res = Result(x, obj, total_newton, float(np.linalg.norm(A @ x - y)))
    res.gap = 2 * n / t  # type: ignore[attr-defined]
    return res


# --------------------------------------------------------------------------- total variation


def tv_admm(A: LinearOperator, y, lam: float, n_iter: int = 200, rho: float = 1.0, cg_iter: int = 30, x0=None):
    """min ½‖Ax−y‖² + λ‖∇x‖₂,₁  by ADMM with the split  z = ∇x  (isotropic TV, images).

    x-update: (AᴴA + ρ∇ᵀ∇) x = Aᴴy + ρ∇ᵀ(z − u), by conjugate gradients.
    z-update: group soft-threshold of ∇x + u with parameter λ/ρ.
    """
    y = np.asarray(y, complex)
    x = np.zeros(A.shape_in, complex) if x0 is None else np.array(x0, complex)
    zx, zy = grad2d(x)
    ux, uy = np.zeros_like(zx), np.zeros_like(zy)
    obj = []
    Ahy = A.H(y)

    def M(v):
        v = v.reshape(A.shape_in)
        gx, gy = grad2d(v)
        return (A.H(A(v)) - rho * div2d(gx, gy)).ravel()

    for _ in range(n_iter):
        rhs = (Ahy - rho * div2d(zx - ux, zy - uy)).ravel()
        x = _cg(M, rhs, cg_iter).reshape(A.shape_in)
        gx, gy = grad2d(x)
        vx, vy = gx + ux, gy + uy
        mag = np.sqrt(np.abs(vx) ** 2 + np.abs(vy) ** 2)
        shrink = np.maximum(1 - (lam / rho) / np.maximum(mag, 1e-30), 0)
        zx, zy = vx * shrink, vy * shrink
        ux, uy = ux + gx - zx, uy + gy - zy
        r = A(x) - y
        obj.append(0.5 * float(np.vdot(r, r).real) + lam * float(np.sqrt(np.abs(gx) ** 2 + np.abs(gy) ** 2).sum()))
    return Result(x, obj, n_iter, float(np.linalg.norm(A(x) - y)))


def tv_fourier(shape, idx, y, lam: float, n_iter: int = 300, rho: float = 1.0, x0=None):
    """min ½‖P F x − y‖² + λ‖∇x‖₂,₁ for partial-Fourier measurements, with an *exact* x-update.

    With periodic finite differences, both PᵀP (a mask) and ∇ᵀ∇ (a Laplacian) are diagonal in the
    Fourier domain, so the ADMM x-update is one division per frequency (Goldstein & Osher 2009, the
    split-Bregman MRI reconstruction). ``idx`` are flat indices into ``fftn(x, norm='ortho')``.
    """
    y = np.asarray(y, complex)
    n1, n2 = shape
    mask = np.zeros(n1 * n2)
    mask[idx] = 1.0
    mask = mask.reshape(shape)
    Y = np.zeros(n1 * n2, complex)
    Y[idx] = y
    Y = Y.reshape(shape)
    # Fourier multipliers of the periodic forward differences
    kx = np.exp(-2j * np.pi * np.fft.fftfreq(n2))[None, :] - 1
    ky = np.exp(-2j * np.pi * np.fft.fftfreq(n1))[:, None] - 1
    denom = mask + rho * (np.abs(kx) ** 2 + np.abs(ky) ** 2)
    denom[denom == 0] = 1.0

    def grad(v):
        return np.roll(v, -1, axis=1) - v, np.roll(v, -1, axis=0) - v

    def div(px, py):  # −gradᵀ for the periodic forward difference above
        return (px - np.roll(px, 1, axis=1)) + (py - np.roll(py, 1, axis=0))

    x = np.zeros(shape, complex) if x0 is None else np.array(x0, complex)
    zx, zy = grad(x)
    ux, uy = np.zeros_like(zx), np.zeros_like(zy)
    obj = []
    for _ in range(n_iter):
        rhs = Y - rho * np.fft.fft2(div(zx - ux, zy - uy), norm="ortho")
        x = np.fft.ifft2(rhs / denom, norm="ortho")
        gx, gy = grad(x)
        vx, vy = gx + ux, gy + uy
        mag = np.sqrt(np.abs(vx) ** 2 + np.abs(vy) ** 2)
        shrink = np.maximum(1 - (lam / rho) / np.maximum(mag, 1e-30), 0)
        zx, zy = vx * shrink, vy * shrink
        ux, uy = ux + gx - zx, uy + gy - zy
        r = np.fft.fft2(x, norm="ortho").ravel()[idx] - y
        obj.append(0.5 * float(np.vdot(r, r).real) + lam * float(mag.sum()))
    return Result(x, obj, n_iter, float(np.linalg.norm(r)))
