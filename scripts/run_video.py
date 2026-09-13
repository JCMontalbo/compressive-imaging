"""Part 4: the 2015 noisy-video paper reproduced (FISTA vs greedy methods) and extended with time
(frame differences) and motion (residual after flow prediction from optical-flow-inverse).

    python scripts/run_video.py [--frames path.npy]

The paper: each frame column x (N px) is measured as y = Φ x with a Gaussian Φ (M x N), the video is
degraded by Gaussian noise σ = 15/255 first, the column is sparse in the DCT, and the recovery is
min ½‖ΦΨᵀs − y‖² + λ‖s‖₁ by FISTA ("FGbCS"), compared with OMP, CoSaMP, IRLS on PSNR against the
clean frame. We use the Sintel shot from optical-flow-inverse (the salesman clip is no longer hosted).
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import dct, idct
from scipy.ndimage import zoom

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from csi import Dense, cosamp, irls, omp  # noqa: E402

FIG = Path(__file__).resolve().parents[1] / "figures"
SIGMA = 15 / 255
N_ROWS = 128


def psnr(a, b):
    return 10 * np.log10(1.0 / max(np.mean((a - b) ** 2), 1e-12))


def load_frames(path, n_frames=12, width=256):
    g = np.load(path)  # [n, H, W] float in [0,1] (from optical-flow-inverse's cache) or uint8
    if g.max() > 1.5:
        g = g / 255.0
    out = []
    for f in g[:n_frames]:
        s = width / f.shape[1]
        r = zoom(f, s, order=1)
        r = r[: N_ROWS] if r.shape[0] >= N_ROWS else np.pad(r, ((0, N_ROWS - r.shape[0]), (0, 0)), mode="edge")
        out.append(np.clip(r, 0, 1))
    return np.stack(out)


def fista_columns(Phi, Y, lam, n_iter=150):
    """FISTA for every column at once: unknown S (N x ncols) DCT coefficients, x = idct(S)."""
    A = Phi @ idct(np.eye(N_ROWS), norm="ortho", axis=0)  # Φ Ψᵀ
    L = np.linalg.norm(A, 2) ** 2
    S = np.zeros((N_ROWS, Y.shape[1]))
    Z, t = S.copy(), 1.0
    for _ in range(n_iter):
        G = A.T @ (A @ Z - Y)
        S_new = Z - G / L
        S_new = np.sign(S_new) * np.maximum(np.abs(S_new) - lam / L, 0)
        t_new = 0.5 * (1 + np.sqrt(1 + 4 * t * t))
        Z = S_new + ((t - 1) / t_new) * (S_new - S)
        S, t = S_new, t_new
    return idct(S, norm="ortho", axis=0)


def greedy_columns(Phi, Y, method, k):
    A = Dense(Phi @ idct(np.eye(N_ROWS), norm="ortho", axis=0))
    X = np.zeros((N_ROWS, Y.shape[1]))
    for j in range(Y.shape[1]):
        if method == "omp":
            s = omp(A, Y[:, j], k).x.real
        elif method == "cosamp":
            s = cosamp(A, Y[:, j], k).x.real
        else:
            s = irls(A, Y[:, j], n_iter=15, cg_iter=60).x.real
        X[:, j] = idct(s, norm="ortho")
    return X


def run(frames, out_path, seed=0):
    rng = np.random.default_rng(seed)
    n_frames, H, W = frames.shape
    noisy = np.clip(frames + SIGMA * rng.standard_normal(frames.shape), 0, 1)
    fracs = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9]
    lam = 0.02  # tuned once on the first frame at M/N = 0.5 (reported)
    results = {"independent (FISTA)": [], "frame differences": [], "motion-compensated residual": []}

    from ofi import horn_schunck_pyramid, propagate_semi_lagrangian  # optical-flow-inverse

    # flows between consecutive *noisy* frames, as an encoder could send them (side information)
    flows = [horn_schunck_pyramid(noisy[k - 1], noisy[k], levels=4) for k in range(1, n_frames)]

    t0 = time.time()
    for fr in fracs:
        M = int(round(fr * N_ROWS))
        Phi = rng.standard_normal((M, N_ROWS)) / np.sqrt(M)
        ind, dif, mot = [], [], []
        prev_d = prev_m = None
        for k in range(n_frames):
            Y = Phi @ noisy[k]
            x_ind = fista_columns(Phi, Y, lam)
            ind.append(psnr(x_ind, frames[k]))
            if k == 0:
                prev_d = prev_m = x_ind
                dif.append(ind[-1])
                mot.append(ind[-1])
                continue
            # D: recover the residual w.r.t. the previous reconstruction
            r = fista_columns(Phi, Y - Phi @ prev_d, lam)
            prev_d = np.clip(prev_d + r, 0, 1)
            dif.append(psnr(prev_d, frames[k]))
            # M: predict along the flow first, then the residual
            u, v = flows[k - 1]
            pred = np.clip(propagate_semi_lagrangian(prev_m, u, v, 1.0), 0, 1)
            r = fista_columns(Phi, Y - Phi @ pred, lam)
            prev_m = np.clip(pred + r, 0, 1)
            mot.append(psnr(prev_m, frames[k]))
        results["independent (FISTA)"].append(np.mean(ind))
        results["frame differences"].append(np.mean(dif))
        results["motion-compensated residual"].append(np.mean(mot))
        print(f"  M/N = {fr:.1f}: independent {np.mean(ind):.2f} dB  differences {np.mean(dif):.2f} dB  motion {np.mean(mot):.2f} dB   ({time.time() - t0:.0f}s)")

    # the paper's comparison at one M/N: a 32-column strip of the first frame (the greedy solvers run
    # column by column and IRLS is slow), all methods on identical measurements
    fr_cmp = 0.9  # the paper: 230 of 256
    M = int(round(fr_cmp * N_ROWS))
    Phi = rng.standard_normal((M, N_ROWS)) / np.sqrt(M)
    cols = slice(W // 2 - 16, W // 2 + 16)
    Y = Phi @ noisy[0][:, cols]
    cmp = {}
    for method in ("FISTA", "omp", "cosamp", "irls"):
        t1 = time.time()
        X = fista_columns(Phi, Y, lam) if method == "FISTA" else greedy_columns(Phi, Y, method, k=24)
        cmp[method] = psnr(X, frames[0][:, cols])
        print(f"  {method:7s} at M/N = {fr_cmp}: {cmp[method]:.2f} dB  ({time.time() - t1:.0f}s)")
    noisy_psnr = psnr(noisy[0][:, cols], frames[0][:, cols])

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.4), constrained_layout=True)
    for name, vals in results.items():
        ax[0].plot(fracs, vals, marker="o", label=name)
    ax[0].axhline(noisy_psnr, c="gray", ls=":", lw=1, label=f"the noisy input itself ({noisy_psnr:.1f} dB)")
    ax[0].set_xlabel("measurements per column / pixels per column (M/N)")
    ax[0].set_ylabel("PSNR vs. clean frame (dB), mean over frames")
    ax[0].set_title(f"Noisy video (σ = 15/255), {n_frames} Sintel frames {H}x{W}, DCT sparsity", fontsize=10)
    ax[0].grid(alpha=0.3)
    ax[0].legend(fontsize=8)
    ax[1].bar(list(cmp.keys()), list(cmp.values()), color=["tab:blue", "tab:gray", "tab:gray", "tab:gray"])
    ax[1].axhline(noisy_psnr, c="gray", ls=":", lw=1)
    ax[1].set_ylabel("PSNR (dB)")
    ax[1].set_title(f"The paper's comparison at M/N = {fr_cmp} (32-column strip, frame 0)", fontsize=10)
    ax[1].set_ylim(min(cmp.values()) - 3, max(cmp.values()) + 2)
    ax[1].grid(alpha=0.3, axis="y")
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return fracs, results, cmp, noisy_psnr


def main():
    ap = argparse.ArgumentParser()
    default = Path(os.environ.get("TMP", "/tmp")) / "bbb" / "sintel_frames.npy"
    ap.add_argument("--frames", type=Path, default=default)
    ap.add_argument("--n-frames", type=int, default=12)
    a = ap.parse_args()
    FIG.mkdir(exist_ok=True)
    frames = load_frames(a.frames, a.n_frames)
    print(f"{len(frames)} frames of {frames.shape[1]}x{frames.shape[2]}")
    fracs, res, cmp, noisy_psnr = run(frames, FIG / "video.png")
    print("\nvideo.png")
    print("  M/N      " + " ".join(f"{f:6.1f}" for f in fracs))
    for name, vals in res.items():
        print(f"  {name:28s}" + " ".join(f"{v:6.2f}" for v in vals))
    # V1: measurements needed for 30 dB (linear interpolation on the curve)
    def need(vals, target=30.0):
        v = np.array(vals)
        if v.max() < target:
            return None
        i = int(np.argmax(v >= target))
        if i == 0:
            return fracs[0]
        return fracs[i - 1] + (target - v[i - 1]) * (fracs[i] - fracs[i - 1]) / (v[i] - v[i - 1])

    print("\n  V1: M/N needed for 30 dB --", ", ".join(f"{k}: {need(v) if need(v) is None else f'{need(v):.2f}'}" for k, v in res.items()))
    print(f"  noisy input PSNR {noisy_psnr:.2f} dB")


if __name__ == "__main__":
    main()
