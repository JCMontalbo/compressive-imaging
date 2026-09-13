import numpy as np

from csi import PartialFourier, basis_pursuit_ip, radial_lines_mask
from csi.solvers import tv_fourier
from csi.thesis import N, cos_sin_dictionary, ftsa, nus_indices, on_bin_tones, shepp_logan, sine_train, synth_from_dictionary


def rel(a, b):
    return np.linalg.norm(a - b) / np.linalg.norm(b)


def test_thesis_recipe_is_not_sparse_and_is_not_recovered():
    f = sine_train([9, 10, 2, 10, 7])
    s = np.sort(np.abs(np.fft.fft(f)))[::-1]
    assert (s[:10] ** 2).sum() / (s**2).sum() < 0.95
    idx = nus_indices(np.random.default_rng(0))
    g = synth_from_dictionary(basis_pursuit_ip(cos_sin_dictionary(N, idx), f[idx]).x, N)
    assert rel(g, f) > 0.1
    assert rel(ftsa(f, 6)[0], f) > 0.1


def test_on_bin_tones_recovered_exactly_from_thesis_sampler():
    f = on_bin_tones([7, 23, 41, 88, 131], 10.0 ** np.arange(1, 6))
    for seed in range(3):
        idx = nus_indices(np.random.default_rng(seed))
        assert len(idx) < 0.25 * N
        g = synth_from_dictionary(basis_pursuit_ip(cos_sin_dictionary(N, idx), f[idx]).x, N)
        assert rel(g, f) < 1e-5


def test_ftsa_fixed_ratio_threshold_fails_on_dynamic_range():
    f = on_bin_tones([7, 23, 41, 88, 131], 10.0 ** np.arange(1, 6))
    h, kept = ftsa(f, 6)
    assert kept < 10 and rel(h, f) > 0.05
    h, kept = ftsa(on_bin_tones([7, 23, 41, 88, 131], np.ones(5)), 6)
    assert kept == 10 and rel(h, f * 0 + on_bin_tones([7, 23, 41, 88, 131], np.ones(5))) < 1e-10


def test_tv_from_radial_lines_beats_zero_fill():
    n = 64
    img = shepp_logan(n)
    idx = radial_lines_mask(n, 22)
    A = PartialFourier((n, n), idx)
    y = A(img)
    rec = tv_fourier((n, n), idx, y, lam=1e-3, rho=0.1, n_iter=300).x.real
    assert rel(rec, img) < 0.1
    assert rel(rec, img) < 0.25 * rel(A.zero_fill(y).real, img)
