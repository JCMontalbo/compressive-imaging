import numpy as np
import pytest

from csi import Dense, basis_pursuit_admm, basis_pursuit_ip, cosamp, fista, gaussian_matrix, irls, ista, omp, tv_admm
from csi.transforms import tv


def sparse_problem(m=64, n=256, k=8, seed=0, noise=0.0):
    rng = np.random.default_rng(seed)
    A = gaussian_matrix(m, n, seed=seed)
    x = np.zeros(n)
    x[rng.choice(n, k, replace=False)] = rng.standard_normal(k)
    y = A(x) + noise * rng.standard_normal(m)
    return A, x, y


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_omp_exact_recovery():
    for seed in range(3):
        A, x, y = sparse_problem(seed=seed)
        assert rel(omp(A, y, 8).x.real, x) < 1e-8


def test_cosamp_exact_and_noise_robust():
    A, x, y = sparse_problem()
    assert rel(cosamp(A, y, 8).x.real, x) < 1e-8
    A, x, yn = sparse_problem(noise=0.01)
    assert rel(cosamp(A, yn, 8).x.real, x) < 0.1  # error proportional to noise, not catastrophic


def test_ista_fista_same_fixed_point_fista_faster():
    A, x, y = sparse_problem()
    lam = 1e-3
    r1 = ista(A, y, lam, n_iter=6000, tol=1e-10)
    r2 = fista(A, y, lam, n_iter=6000, tol=1e-10)
    # ISTA is O(1/k): after 6000 iterations it is still ~1% above the FISTA objective, and monotone
    assert np.all(np.diff(r1.objective) <= 1e-12)
    assert r1.objective[-1] >= r2.objective[-1] - 1e-12
    assert abs(r1.objective[-1] - r2.objective[-1]) < 2e-2 * r2.objective[-1]
    assert r2.iters < r1.iters
    assert rel(r2.x.real, x) < 1e-2  # lasso bias only


def test_fista_rate_bound():
    """Beck–Teboulle Thm 4.4: F(x_k) − F* ≤ 2L‖x0 − x*‖² / (k+1)²."""
    A, x, y = sparse_problem()
    lam = 1e-3
    L = A.norm_sq()
    star = fista(A, y, lam, n_iter=20000, tol=1e-14)
    Fstar = star.objective[-1]
    r = fista(A, y, lam, n_iter=300, tol=0)
    d0 = np.linalg.norm(star.x) ** 2
    for k, Fk in enumerate(r.objective, start=1):
        assert Fk - Fstar <= 2 * L * d0 / (k + 1) ** 2 + 1e-12


def test_irls_admm_ip_agree_on_basis_pursuit():
    A, x, y = sparse_problem()
    xi = irls(A, y).x.real
    xa = basis_pursuit_admm(A, y).x.real
    xp = basis_pursuit_ip(A.A, y)
    assert rel(xp.x, x) < 1e-5
    assert np.linalg.norm(xp.x - xa) < 1e-4
    assert np.linalg.norm(xp.x - xi) < 1e-4


def test_interior_point_kkt():
    A, x, y = sparse_problem()
    r = basis_pursuit_ip(A.A, y, tol=1e-9)
    assert r.gap < 1e-8  # duality gap
    assert r.residual < 1e-6  # equality constraint
    assert abs(np.abs(r.x).sum() - np.abs(x).sum()) < 1e-5  # ℓ1 optimum equals the sparse truth here


def test_tv_admm_recovers_piecewise_constant():
    rng = np.random.default_rng(0)
    n = 32
    img = np.zeros((n, n))
    img[8:24, 8:24] = 1.0
    img[12:20, 12:20] = 0.5
    A = Dense(np.eye(n * n))  # denoising: identity measurement

    class Id:
        shape_in = shape_out = (n, n)

        def __call__(self, x):
            return x

        def H(self, y):
            return y

        def norm_sq(self):
            return 1.0

    noise = 0.1 * rng.standard_normal((n, n))
    r = tv_admm(Id(), img + noise, lam=0.15, n_iter=100)
    assert np.linalg.norm(r.x.real - img) < 0.5 * np.linalg.norm(noise)
    assert tv(r.x.real) < tv(img + noise)
