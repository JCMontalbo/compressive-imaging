"""The README's lead figure: keep less, see how much survives.

Top row: a natural image rebuilt from its k largest Fourier coefficients as k sweeps from 0.05 % to 30 %
(the thesis's idea), with the kept coefficients shown in k-space. Bottom row: the ISAR target rebuilt from a
growing random fraction of its phase-history samples, zero-filled (left) and by image-domain l1 (right).

    python scripts/make_hero.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi.isar import Radar, image_from_rect, keep_random, phase_history, pipeline_image, pipeline_zero_fill, polar_to_rect, pseudo_aircraft, render_scene  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"


def db(img):
    """Linear amplitude, normalised: the sidelobes stay invisible, the scatterers do not."""
    a = np.abs(img)
    return a / a.max()


def main():
    frame_path = Path(os.environ.get("TMP", "/tmp")) / "bbb" / "sintel_frames.npy"
    img = np.load(frame_path)[0]
    F = np.fft.fft2(img)
    order = np.argsort(np.abs(F).ravel())[::-1]
    fracs = np.concatenate([np.logspace(np.log10(0.0005), np.log10(0.3), 22), [0.3] * 4])
    fracs = np.concatenate([fracs, fracs[::-1][1:-1]])  # sweep up, hold, sweep back

    radar, scene = Radar(), pseudo_aircraft()
    rect = polar_to_rect(phase_history(scene, radar), radar, 64)
    up = np.concatenate([np.logspace(np.log10(0.02), np.log10(0.4), 22), [0.4] * 4])
    isar_fracs = np.concatenate([up, up[::-1][1:-1]])  # 2 % .. 40 % of the samples, same timing as the top row
    cache = {}

    def coded(frac):
        keep = np.zeros(F.size, bool)
        keep[order[: max(1, int(frac * F.size))]] = True
        rec = np.fft.ifft2((F.ravel() * keep).reshape(F.shape)).real
        mask = np.fft.fftshift(keep.reshape(F.shape))
        err = np.linalg.norm(rec - img) / np.linalg.norm(img)
        return rec, mask, err

    def isar(frac):
        key = round(float(frac), 3)
        if key not in cache:
            idx = keep_random(rect.size, key, seed=0)
            cache[key] = (pipeline_zero_fill(rect, idx), pipeline_image(rect, idx, n_iter=300))
        return cache[key]

    truth = render_scene(scene, radar, 64)
    fig = plt.figure(figsize=(12, 6.6))
    gs = fig.add_gridspec(2, 4, width_ratios=[1.9, 1, 1, 1], height_ratios=[1, 1])
    a_img = fig.add_subplot(gs[0, 0:2])
    a_mask = fig.add_subplot(gs[0, 2])
    a_txt = fig.add_subplot(gs[0, 3])
    a_txt2 = fig.add_subplot(gs[1, 0])
    a_truth = fig.add_subplot(gs[1, 1])
    a_zf = fig.add_subplot(gs[1, 2])
    a_l1 = fig.add_subplot(gs[1, 3])
    for a in (a_img, a_mask, a_truth, a_zf, a_l1, a_txt, a_txt2):
        a.set_xticks([])
        a.set_yticks([])
    for a in (a_txt, a_txt2):
        a.axis("off")
    a_truth.imshow(np.abs(truth), cmap="gray_r")
    a_truth.set_title("the target: 39 point scatterers", fontsize=10)

    rec, mask, err = coded(fracs[0])
    im_img = a_img.imshow(rec, cmap="gray", vmin=0, vmax=1)
    im_mask = a_mask.imshow(mask, cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    zf, l1 = isar(isar_fracs[0])
    im_zf = a_zf.imshow(db(zf), cmap="gray_r", vmin=0, vmax=1)
    im_l1 = a_l1.imshow(db(l1), cmap="gray_r", vmin=0, vmax=1)
    a_mask.set_title("the Fourier coefficients kept", fontsize=10)
    a_zf.set_title("zero-filled", fontsize=10)
    a_l1.set_title("l1, sparsity in the image", fontsize=10)
    t1 = a_txt.text(0.0, 0.5, "", fontsize=11, va="center", ha="left", transform=a_txt.transAxes)
    t2 = a_txt2.text(0.02, 0.5, "", fontsize=11, va="center", ha="left", transform=a_txt2.transAxes)
    a_txt2.set_title("", fontsize=10)
    ttl = a_img.set_title("", fontsize=10)
    fig.tight_layout()

    def update(i):
        f = fracs[i]
        rec, mask, err = coded(f)
        im_img.set_data(rec)
        im_mask.set_data(mask)
        ttl.set_text(f"a natural image from {f:.2%} of its Fourier coefficients")
        t1.set_text(f"Keep the largest Fourier\ncoefficients, zero the rest,\ninvert.\n\nkept: {f:.2%}\nerror: {err:.1%}")
        g = isar_fracs[i]
        zf, l1 = isar(g)
        im_zf.set_data(db(zf))
        im_l1.set_data(db(l1))
        t2.set_text(f"Radar: a 39-scatterer target\nfrom {g:.0%} of the phase-\nhistory samples.\n\nSame samples, two answers:\nsparsity assumed in the data\n(zero-fill) or in the image (l1).")
        return [im_img, im_mask, im_zf, im_l1, t1, t2, ttl]

    anim = FuncAnimation(fig, update, frames=len(fracs), interval=1000 / 6, blit=False)
    anim.save(FIG / "hero.gif", writer=PillowWriter(fps=6), dpi=90)
    # a still at 5 % / 22.5 %
    update(int(np.argmin(np.abs(fracs[: len(fracs) // 2] - 0.05))))
    fig.savefig(FIG / "hero.png", dpi=110)
    plt.close(fig)
    print("wrote figures/hero.gif, figures/hero.png")


if __name__ == "__main__":
    main()
