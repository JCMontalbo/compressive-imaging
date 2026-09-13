"""Part 2: the master's thesis reproduced and labelled honestly (docs/plan.md).

    python scripts/run_thesis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi import PartialFourier, basis_pursuit_ip, radial_lines_mask  # noqa: E402
from csi.solvers import tv_fourier  # noqa: E402
from csi.thesis import N, T, chirp_train, cos_sin_dictionary, ftsa, nus_indices, on_bin_tones, shepp_logan, sine_train, synth_from_dictionary, wht_tsa  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"


def rel(a, b):
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def l1_from_samples(f, idx):
    D = cos_sin_dictionary(N, idx)
    r = basis_pursuit_ip(D, f[idx])
    return synth_from_dictionary(r.x, N)


def fig_example1():
    """The thesis's Example 1 (rᵢ = 9, 10, 2, 10, 7), as written: FTSA and l1 from its sampler."""
    f = sine_train([9, 10, 2, 10, 7])
    idx = nus_indices(np.random.default_rng(0))
    h, nk = ftsa(f, 6)
    g = l1_from_samples(f, idx)
    F = np.abs(np.fft.fft(f))
    s = np.sort(F)[::-1]
    top10 = float((s[:10] ** 2).sum() / (s**2).sum())
    fig, ax = plt.subplots(2, 2, figsize=(13, 6.5), constrained_layout=True)
    ax[0, 0].plot(T, f, lw=0.6)
    ax[0, 0].plot(T[idx], f[idx], "r.", ms=4)
    ax[0, 0].set_title(f"Example 1 as written: 5 tones, 400 samples, {len(idx)} non-uniform samples (red)", fontsize=10)
    ax[0, 1].semilogy(np.fft.fftshift(F) + 1e-3, lw=0.6)
    ax[0, 1].set_title(f"|DFT|: energy in the 10 largest coefficients = {top10:.3f}  (10-sparse would be 1.000)", fontsize=10)
    ax[1, 0].plot(T, f, lw=0.6, label="signal")
    ax[1, 0].plot(T, h, lw=0.6, label=f"FTSA, K = 6 ({nk} coefficients kept, sees the whole signal): rel. error {rel(h, f):.2f}")
    ax[1, 0].legend(fontsize=8)
    ax[1, 1].plot(T, f, lw=0.6, label="signal")
    ax[1, 1].plot(T, g, lw=0.6, label=f"l1 (interior point) from the {len(idx)} samples: rel. error {rel(g, f):.2f}")
    ax[1, 1].legend(fontsize=8)
    for a in ax.ravel():
        a.grid(alpha=0.3)
    fig.suptitle("The thesis's signal is not sparse on the grid (aliased, off-bin tones), so neither method recovers it", fontsize=11)
    fig.savefig(FIG / "thesis_example1.png", dpi=110)
    plt.close(fig)
    return dict(top10=top10, ftsa_err=rel(h, f), ftsa_kept=nk, l1_err=rel(g, f), n_samples=len(idx))


def fig_sampling_curve(draws=20):
    """Sampling fraction vs. error for on-bin five-tone signals with the thesis's amplitudes and equal ones."""
    rng = np.random.default_rng(1)
    bins = rng.choice(np.arange(3, 150), 5, replace=False)
    signals = {
        "on-bin, thesis amplitudes 10..10^5": on_bin_tones(bins, 10.0 ** np.arange(1, 6)),
        "on-bin, equal amplitudes": on_bin_tones(bins, np.ones(5)),
        "thesis recipe (aliased)": sine_train([9, 10, 2, 10, 7]),
    }
    steps = [2, 3, 4, 5, 6, 8, 10, 14, 20]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    table = {}
    for name, f in signals.items():
        fr, med, lo, hi, frac_ok = [], [], [], [], []
        for st in steps:
            errs, ns = [], []
            for d in range(draws):
                idx = nus_indices(np.random.default_rng(100 * st + d), max_step=st)
                errs.append(rel(l1_from_samples(f, idx), f))
                ns.append(len(idx))
            errs = np.array(errs)
            fr.append(np.mean(ns) / N)
            med.append(np.median(errs))
            lo.append(np.percentile(errs, 10))
            hi.append(np.percentile(errs, 90))
            frac_ok.append(np.mean(errs < 1e-3))
        table[name] = (fr, med, frac_ok)
        ax[0].semilogy(fr, med, marker="o", label=name)
        ax[0].fill_between(fr, lo, hi, alpha=0.15)
        ax[1].plot(fr, frac_ok, marker="o", label=name)
    for a in ax:
        a.axvline(67 / N, c="gray", ls="--", lw=1)
        a.grid(alpha=0.3)
        a.set_xlabel("fraction of the 400 samples kept (thesis sampler, varying step)")
        a.legend(fontsize=8)
    ax[0].text(67 / N + 0.005, 2e-6, "thesis: 67 samples", fontsize=8, color="gray")
    ax[0].set_ylabel("relative error of l1 recovery (median, 10-90 %)")
    ax[0].set_title("l1 from non-uniform samples, interior point, 20 draws per point")
    ax[1].set_ylabel("fraction of draws with error < 1e-3")
    ax[1].set_title("claim T1: >= 90 % of draws below 1e-3 at 17 % sampling?")
    ax[1].set_ylim(-0.05, 1.05)
    fig.savefig(FIG / "thesis_sampling.png", dpi=110)
    plt.close(fig)
    # FTSA on the same signals, for the record
    ftsa_rows = {name: (rel(ftsa(f, 6)[0], f), ftsa(f, 6)[1]) for name, f in signals.items()}
    return table, ftsa_rows


def chirps_and_wht():
    c = chirp_train([9, 10, 2, 10, 7])
    idx = nus_indices(np.random.default_rng(0))
    h, nk = ftsa(c, 2)
    g = l1_from_samples(c, idx)
    tones = on_bin_tones([7, 23, 41, 88, 131], np.ones(5))
    hw, nw = wht_tsa(tones, 6)
    hf, nf = ftsa(tones, 6)
    return dict(chirp_ftsa=(rel(h, c), nk), chirp_l1=rel(g, c), tones_wht=(rel(hw, tones), nw), tones_dft=(rel(hf, tones), nf))


def fig_radial():
    n = 128
    img = shepp_logan(n)
    lines = [6, 10, 14, 18, 22, 30, 40, 60]
    errs_tv, errs_zf, fracs = [], [], []
    for L in lines:
        idx = radial_lines_mask(n, L)
        A = PartialFourier((n, n), idx)
        y = A(img)
        rec = tv_fourier((n, n), idx, y, lam=1e-3, rho=0.1, n_iter=400).x.real
        errs_tv.append(rel(rec, img))
        errs_zf.append(rel(A.zero_fill(y).real, img))
        fracs.append(len(idx) / n**2)
    fig = plt.figure(figsize=(14, 4.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.6, 1, 1, 1])
    a = fig.add_subplot(gs[0, 0])
    a.semilogy(lines, errs_zf, marker="s", c="gray", label="zero-filled inverse FFT")
    a.semilogy(lines, errs_tv, marker="o", label="TV minimisation (ADMM, exact Fourier x-update)")
    a.axvline(40, c="gray", ls="--", lw=1)
    a.text(40.5, 0.5, "thesis: 40 lines", fontsize=8, color="gray")
    a.set_xlabel("radial lines in k-space")
    a.set_ylabel("relative error")
    a.set_title("Shepp-Logan 128 x 128 from radial Fourier lines", fontsize=10)
    a.grid(alpha=0.3)
    a.legend(fontsize=8)
    idx22, idx40 = radial_lines_mask(n, 22), radial_lines_mask(n, 40)
    panels = [
        ("phantom", img),
        (f"zero-fill, 22 lines ({len(idx22) / n**2:.0%} of k-space)", PartialFourier((n, n), idx22).zero_fill(PartialFourier((n, n), idx22)(img)).real),
        ("TV, 22 lines", tv_fourier((n, n), idx22, PartialFourier((n, n), idx22)(img), lam=1e-3, rho=0.1, n_iter=400).x.real),
    ]
    for j, (t, im) in enumerate(panels, start=1):
        ax = fig.add_subplot(gs[0, j])
        ax.imshow(im, cmap="gray", vmin=0, vmax=1)
        ax.set_title(t, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(FIG / "thesis_radial.png", dpi=110)
    plt.close(fig)
    return lines, fracs, errs_zf, errs_tv


def main():
    FIG.mkdir(exist_ok=True)
    e1 = fig_example1()
    print("thesis_example1.png")
    print(f"  energy in top-10 DFT coefficients {e1['top10']:.3f}; FTSA (K=6, {e1['ftsa_kept']} kept) rel err {e1['ftsa_err']:.3f}; "
          f"l1 from {e1['n_samples']} samples rel err {e1['l1_err']:.3f}")

    table, ftsa_rows = fig_sampling_curve()
    print("\nthesis_sampling.png -- l1 from the thesis's sampler (interior point), median rel err / fraction < 1e-3")
    for name, (fr, med, ok) in table.items():
        print(f"  {name}")
        print("    fraction " + " ".join(f"{x:6.2f}" for x in fr))
        print("    median   " + " ".join(f"{x:6.0e}" for x in med))
        print("    frac<1e-3" + " ".join(f"{x:6.2f}" for x in ok))
    print("  FTSA K=6 on the same signals (sees the whole signal):")
    for name, (err, kept) in ftsa_rows.items():
        print(f"    {name:36s} rel err {err:.3f} ({kept} coefficients)")

    cw = chirps_and_wht()
    print("\nchirps and Walsh-Hadamard")
    print(f"  chirp train: FTSA K=2 rel err {cw['chirp_ftsa'][0]:.3f} ({cw['chirp_ftsa'][1]} coefficients), l1 from samples rel err {cw['chirp_l1']:.3f}")
    print(f"  on-bin tones: DFT thresholding rel err {cw['tones_dft'][0]:.2e} ({cw['tones_dft'][1]} coeffs) vs WHT thresholding {cw['tones_wht'][0]:.3f} ({cw['tones_wht'][1]} coeffs)")

    lines, fracs, ezf, etv = fig_radial()
    print("\nthesis_radial.png -- Shepp-Logan from radial lines")
    for L, f, a, b in zip(lines, fracs, ezf, etv):
        print(f"  {L:3d} lines ({f:5.1%} of k-space): zero-fill {a:.3f}  TV {b:.2e}")


if __name__ == "__main__":
    main()
