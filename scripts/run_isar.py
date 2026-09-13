"""Part 3: the 2016 ISAR paper rebuilt, its failure reproduced and diagnosed (docs/plan.md).

    python scripts/run_isar.py

Writes figures/isar_*.png and prints every number quoted in the README.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi.isar import (  # noqa: E402
    Radar,
    Scene,
    f1_score,
    image_from_rect,
    keep_random,
    phase_history,
    pipeline_image,
    pipeline_paper,
    pipeline_zero_fill,
    polar_to_rect,
    pseudo_aircraft,
    render_scene,
    sparsity_energy,
)

FIG = Path(__file__).resolve().parents[1] / "figures"
N = 64
FRACTION = 0.225  # the paper: 45 % of Nyquist


def db(img):
    a = np.abs(img)
    return 20 * np.log10(a / a.max() + 1e-9)


def energy_on_support(img, truth):
    m = np.abs(truth) > 0
    a = np.abs(img) ** 2
    return float(a[m].sum() / a.sum())


def add_noise(rect, snr_db, seed):
    rng = np.random.default_rng(seed)
    p = np.mean(np.abs(rect) ** 2)
    s = np.sqrt(p / 10 ** (snr_db / 10) / 2)
    return rect + s * (rng.standard_normal(rect.shape) + 1j * rng.standard_normal(rect.shape))


def _clean(ax, t=None):
    ax.set_xticks([])
    ax.set_yticks([])
    if t:
        ax.set_title(t, fontsize=9)


def fig_pipelines(radar, scene):
    rect = polar_to_rect(phase_history(scene, radar), radar, N)
    truth = render_scene(scene, radar, N)
    idx = keep_random(rect.size, FRACTION, seed=0)
    rows = {}
    for snr in (None, 20.0):
        r = rect if snr is None else add_noise(rect, snr, 1)
        rows[snr] = {
            "full data": image_from_rect(r),
            "zero-fill": pipeline_zero_fill(r, idx),
            "paper (l1 on data)": pipeline_paper(r, idx),
            "image-domain l1": pipeline_image(r, idx),
        }
    fig, ax = plt.subplots(2, 5, figsize=(17, 7), constrained_layout=True)
    for i, snr in enumerate((None, 20.0)):
        ax[i, 0].imshow(np.abs(truth), cmap="gray_r")
        _clean(ax[i, 0], "scatterers (truth)" if i == 0 else "")
        for j, (name, img) in enumerate(rows[snr].items(), start=1):
            f1, err = f1_score(img, truth)
            ax[i, j].imshow(db(img), cmap="gray_r", vmin=-30, vmax=0)
            _clean(ax[i, j], f"{name}{'' if name == 'full data' else f' @ {FRACTION:.1%}'}\nF1 {f1:.2f}, energy on target {energy_on_support(img, truth):.2f}")
        ax[i, 0].set_ylabel("noise-free" if snr is None else f"SNR {snr:.0f} dB", fontsize=10)
    fig.suptitle("ISAR pseudo-aircraft, 22.5 % of the phase history kept: the paper's pipeline vs. image-domain l1 (dB, 30 dB range)", fontsize=11)
    fig.savefig(FIG / "isar_pipelines.png", dpi=110)
    plt.close(fig)
    out = {}
    for snr, d in rows.items():
        out[snr] = {k: (f1_score(v, truth), energy_on_support(v, truth)) for k, v in d.items()}
    h3 = dict(data=sparsity_energy(rect, scene.n), image=sparsity_energy(truth, scene.n), full_image=sparsity_energy(rows[None]["full data"], scene.n))
    return out, h3


def fig_phase_transition(radar, trials=8):
    fractions = np.array([0.02, 0.05, 0.1, 0.15, 0.2, 0.225, 0.3, 0.4, 0.5])
    counts = [10, 20, 40, 80, 160, 320, 640]
    P = np.zeros((len(counts), len(fractions)))
    rng = np.random.default_rng(0)
    t0 = time.time()
    for a, k in enumerate(counts):
        for b, fr in enumerate(fractions):
            ok = 0
            for t in range(trials):
                d = radar.range_res
                span = (N // 2 - 4) * d
                scene = Scene(rng.uniform(-span, span, k), rng.uniform(-span, span, k), np.ones(k))
                truth = render_scene(scene, radar, N)
                rect = np.fft.fft2(np.fft.ifftshift(truth), norm="ortho")
                idx = keep_random(rect.size, fr, seed=1000 * a + 10 * b + t)
                img = pipeline_image(rect, idx, n_iter=200)
                ok += f1_score(img, truth)[0] >= 0.9
            P[a, b] = ok / trials
        print(f"  phase transition: {k} scatterers done ({time.time() - t0:.0f}s)")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.imshow(P, origin="lower", aspect="auto", cmap="viridis", vmin=0, vmax=1, extent=[fractions[0], fractions[-1], -0.5, len(counts) - 0.5])
    ax.set_yticks(range(len(counts)))
    ax.set_yticklabels(counts)
    ax.set_xticks(fractions)
    ax.set_xticklabels([f"{f:.2f}" for f in fractions], fontsize=7)
    ax.axvline(FRACTION, c="w", ls="--", lw=1)
    ax.text(FRACTION + 0.005, len(counts) - 0.9, "paper: 22.5 %", color="w", fontsize=8)
    ax.set_xlabel("fraction of phase-history samples kept")
    ax.set_ylabel("number of scatterers (64 x 64 image)")
    ax.set_title("Image-domain l1: probability of F1 >= 0.9 (8 random scenes per cell)")
    fig.colorbar(im, ax=ax, label="P(success)")
    fig.tight_layout()
    fig.savefig(FIG / "isar_phase_transition.png", dpi=110)
    plt.close(fig)
    return fractions, counts, P


def fig_resolution(radar, trials=6, oversample=2):
    """Two equal scatterers separated by s Fourier cells along range. The image grid is oversampled
    (pixel = cell / oversample) and only the in-band k-samples are measured, so resolving below one
    cell is a genuine super-resolution question, not a grid artefact."""
    from csi.isar import _ImageMeasure
    from csi.solvers import fista

    seps = [0.4, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0]
    snrs = [10, 20, 30, 40]
    res = np.zeros((len(snrs), len(seps)))
    d = radar.range_res
    n = N * oversample
    # in-band k indices of the oversampled grid: the central N x N block (fftshift convention)
    band = np.zeros((n, n), bool)
    lo = n // 2 - N // 2
    band[lo : lo + N, lo : lo + N] = True
    band_idx = np.flatnonzero(np.fft.ifftshift(band).ravel())
    for a, snr in enumerate(snrs):
        for b, sep in enumerate(seps):
            ok = 0
            for t in range(trials):
                rng = np.random.default_rng(1000 * a + 10 * b + t)
                # exact image at fine resolution from the two scatterers, band-limited measurement
                xs = np.array([-sep * d / 2, sep * d / 2])
                pix = d / oversample
                truth = np.zeros((n, n), complex)
                cols = np.round(xs / pix).astype(int) + n // 2
                truth[n // 2, cols[0]] += 1
                truth[n // 2, cols[1]] += 1
                data = np.fft.fft2(np.fft.ifftshift(truth), norm="ortho")
                keep = rng.choice(band_idx, int(round(FRACTION * len(band_idx))), replace=False)
                y = data.ravel()[keep]
                p = np.mean(np.abs(y) ** 2)
                sig = np.sqrt(p / 10 ** (snr / 10) / 2)
                y = y + sig * (rng.standard_normal(len(y)) + 1j * rng.standard_normal(len(y)))
                A = _ImageMeasure((n, n), keep)
                img = fista(A, y, 0.05 * np.abs(A.H(y)).max(), n_iter=400).x
                n_true = int((np.abs(truth) > 0).sum())
                f1, _ = f1_score(img, truth, tol_cells=0.5)
                ok += (n_true == 2) and (f1 >= 0.99)
            res[a, b] = ok / trials
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for a, snr in enumerate(snrs):
        ax.plot(seps, res[a], marker="o", label=f"SNR {snr} dB")
    ax.axvline(1.0, c="gray", ls="--", lw=1)
    ax.text(1.02, 0.05, "Fourier resolution cell", fontsize=8, color="gray")
    ax.set_xlabel("separation of two scatterers (Fourier resolution cells)")
    ax.set_ylabel("fraction of trials with both resolved")
    ax.set_title(f"Two-scatterer resolution, image-domain l1, {FRACTION:.1%} of in-band samples, {oversample}x grid")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "isar_resolution.png", dpi=110)
    plt.close(fig)
    return seps, snrs, res


def fig_motion(radar, scene):
    """Translational range walk smears the image; compensating it on the kept samples works for the
    image-domain pipeline, and does nothing for the zero-filled one (the paper's 'disaster')."""
    truth = render_scene(scene, radar, N)
    D = phase_history(scene, radar)
    # translational motion: range changes linearly over the CPI by 3 resolution cells
    walk = np.linspace(0, 3 * radar.range_res, radar.n_pulse)
    phase = np.exp(-1j * 4 * np.pi * radar.freqs[None, :] * walk[:, None] / 299_792_458.0)
    D_moving = D * phase
    rect_m = polar_to_rect(D_moving, radar, N)
    rect_c = polar_to_rect(D_moving * np.conj(phase), radar, N)  # compensated with the known walk
    idx = keep_random(rect_m.size, FRACTION, seed=0)
    panels = [
        ("full data, uncompensated", image_from_rect(rect_m)),
        ("full data, compensated", image_from_rect(rect_c)),
        ("zero-fill 22.5 %, compensated", pipeline_zero_fill(rect_c, idx)),
        ("image-domain l1 22.5 %, compensated", pipeline_image(rect_c, idx)),
    ]
    fig, ax = plt.subplots(1, 5, figsize=(17, 3.8), constrained_layout=True)
    ax[0].imshow(np.abs(truth), cmap="gray_r")
    _clean(ax[0], "truth")
    out = {}
    for a, (name, img) in zip(ax[1:], panels):
        f1, _ = f1_score(img, truth)
        out[name] = (f1, energy_on_support(img, truth))
        a.imshow(db(img), cmap="gray_r", vmin=-30, vmax=0)
        _clean(a, f"{name}\nF1 {f1:.2f}, energy on target {out[name][1]:.2f}")
    fig.suptitle("Translational motion (3-cell range walk over the CPI) and its compensation", fontsize=11)
    fig.savefig(FIG / "isar_motion.png", dpi=110)
    plt.close(fig)
    return out


def main():
    FIG.mkdir(exist_ok=True)
    radar, scene = Radar(), pseudo_aircraft()
    print(f"radar: {radar.f0 / 1e9:.3f}-{(radar.f0 + radar.bandwidth) / 1e9:.3f} GHz, range cell {radar.range_res:.3f} m, "
          f"aperture {radar.dtheta:.4f} rad ({radar.cpi_seconds:.2f} s at {radar.omega} rad/s), {radar.n_pulse} pulses x {radar.n_freq} frequencies; "
          f"target: {scene.n} scatterers; image {N}x{N}")

    out, h3 = fig_pipelines(radar, scene)
    print("\nisar_pipelines.png")
    for snr, d in out.items():
        print(f"  {'noise-free' if snr is None else f'SNR {snr:.0f} dB'}:")
        for k, ((f1, err), e) in d.items():
            print(f"    {k:22s} F1 {f1:.3f}  loc err {err:.2f} cells  energy on target {e:.3f}")
    print(f"  H3: energy in best-{scene.n} terms -- data (sampling basis) {h3['data']:.3f}, rendered image {h3['image']:.3f}, full-data image {h3['full_image']:.3f}")

    m = fig_motion(radar, scene)
    print("\nisar_motion.png")
    for k, (f1, e) in m.items():
        print(f"  {k:36s} F1 {f1:.3f}  energy on target {e:.3f}")

    fr, counts, P = fig_phase_transition(radar)
    print("\nisar_phase_transition.png -- P(F1 >= 0.9), rows = scatterers, cols = fraction kept")
    print("        " + " ".join(f"{f:5.2f}" for f in fr))
    for k, row in zip(counts, P):
        print(f"  {k:4d}  " + " ".join(f"{p:5.2f}" for p in row))

    seps, snrs, R = fig_resolution(radar)
    print("\nisar_resolution.png -- fraction resolved, rows = SNR, cols = separation (cells)")
    print("        " + " ".join(f"{s:5.2f}" for s in seps))
    for s, row in zip(snrs, R):
        print(f"  {s:4d}  " + " ".join(f"{p:5.2f}" for p in row))


if __name__ == "__main__":
    main()
