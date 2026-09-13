import numpy as np

from csi import Compose, Dense, InverseDCT, InverseWHT, NonUniformSamples, PartialFourier, fwht, gaussian_matrix, radial_lines_mask, random_fourier_mask


def test_adjoints():
    rng = np.random.default_rng(0)
    ops = [
        gaussian_matrix(40, 128),
        PartialFourier((32, 32), random_fourier_mask((32, 32), 0.3)),
        PartialFourier((64,), random_fourier_mask((64,), 0.5)),
        NonUniformSamples(200, rng.choice(200, 40, replace=False)),
        InverseDCT((16, 16)),
        InverseWHT(128),
        Compose(gaussian_matrix(30, 64), InverseDCT((64,))),
    ]
    for op in ops:
        assert op.adjoint_check() < 1e-10, type(op).__name__


def test_norm_bound_is_upper_bound():
    A = gaussian_matrix(40, 128, seed=3)
    true = np.linalg.norm(A.A, 2) ** 2
    est = A.norm_sq()
    assert true <= est <= 1.2 * true


def test_partial_fourier_zero_fill_is_adjoint():
    idx = random_fourier_mask((16, 16), 0.4)
    A = PartialFourier((16, 16), idx)
    x = np.random.default_rng(1).standard_normal((16, 16))
    assert np.allclose(A.zero_fill(A(x)), A.H(A(x)))


def test_wht_is_orthonormal_involution():
    a = np.random.default_rng(2).standard_normal(256)
    assert np.allclose(fwht(fwht(a)), a)
    assert np.isclose(np.linalg.norm(fwht(a)), np.linalg.norm(a))


def test_radial_lines_count_grows():
    assert len(radial_lines_mask(64, 10)) < len(radial_lines_mask(64, 40)) < 64 * 64
