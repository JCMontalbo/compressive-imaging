# compressive-imaging

[![tests](https://github.com/JCMontalbo/compressive-imaging/actions/workflows/ci.yml/badge.svg)](https://github.com/JCMontalbo/compressive-imaging/actions/workflows/ci.yml)

My master's work (UTRGV, 2016) and two SPIE papers looked at sparse approximation and ℓ₁ recovery for
radar signals and noisy video, leaning on off-the-shelf solvers and toy signals. The ISAR paper's
reconstruction did not work, and said so. The thesis's last sentence named the next step: *"learn more
about interior point algorithms so as to try and develop our own compressive sensing algorithm that can
be utilized within the radar system process."*

This repository is that next step, done with the pass/fail lines written down first
([docs/plan.md](docs/plan.md)):

1. **The solvers, written from scratch** and tested against theory — ISTA, FISTA, OMP, CoSaMP, IRLS,
   ADMM basis pursuit, a log-barrier interior-point method for basis pursuit, and ADMM total variation
   with an exact Fourier-domain update.
2. **The thesis's experiments reproduced and labelled honestly** — including the finding that its test
   signals were never sparse, which is why its ℓ₁ step "needed the original signal".
3. **The 2016 ISAR model rebuilt, the failure reproduced, and the reason found** — the paper applied
   sparsity to the phase history; the *image* is what is sparse. Same 22.5 % of the samples, F1 0.56 → 1.00.
4. **The 2015 video method reproduced and extended** with frame differences and motion-compensated
   residuals using the flow from [optical-flow-inverse](https://github.com/JCMontalbo/optical-flow-inverse).

Every number below comes from a script in `scripts/`; 20 tests, CI on Python 3.10–3.13.

Sources: J. Montalbo, *Compressive Sensing and Radar Imaging*, MS thesis, UTRGV 2016 ·
Hu, Montalbo, Li, Sun, Qiao, *Sparse representation for the ISAR image reconstruction*, Proc. SPIE 9857 (2016) ·
Zhao, Montalbo, Li, Sun, Qiao, *Compressive sensing for noisy video reconstruction*, Proc. SPIE 9484 (2015).

---

## 3. The ISAR paper: why the 2016 reconstruction failed, and the fix

The paper's model, built properly ([`csi/isar.py`](csi/isar.py)): far-field Born phase history of point
scatterers on a turntable, stepped-frequency 3.0–3.384 GHz (0.39 m range cell), 64 pulses over the aperture
that makes cross-range cells the same size, the paper's pseudo-aircraft of 37 scatterers, polar→rectangular
interpolation, image by 2-D FFT. Compressive sampling as in the paper: keep a uniformly random 22.5 % of the
phase-history samples ("45 % of Nyquist").

![isar pipelines](figures/isar_pipelines.png)

| pipeline, 22.5 % of samples | F1 (peaks within one cell of a true scatterer) | energy on the true scatterers |
|---|---|---|
| full data (reference, the paper's Fig. 3) | 1.00 | 0.76 |
| zero-filled inverse FFT | 0.69 | **0.19** |
| the paper's: ℓ₁ applied to the *data*, then image (Fig. 4) | 0.56 | 0.50 |
| **ℓ₁ applied to the *image*** (same samples) | **1.00** | **0.82** |
| … at 20 dB SNR | 1.00 | 0.82 |

**What went wrong in 2016.** The paper's Section 3 finds "the most sparse representation of the original
signal" — the phase history — and then images it. But the phase history is a sum of 37 complex exponentials
and is not sparse: its 37 largest samples hold **21.6 %** of its energy. The image is exactly 37-sparse
(100 %). Sparsity was assumed in the wrong domain. In the physically meaningful reading (a random subset of
the *data* is what was acquired), the sparsest data consistent with the samples is simply the samples with
zeros elsewhere, and the method degenerates to zero-filling — whose aliasing noise is the paper's Figure 4.
Treating the kept samples as partial Fourier measurements of the image and solving for the image gives the
target back at the same sampling rate, with or without noise.

**Against the pre-registered criteria:** H2 (image-domain ℓ₁ at 22.5 %: F1 ≥ 0.9, localisation < 1 cell)
**supported**, noise-free and at 20 dB. H3 (data not sparse in the basis used, image is) **supported**.
H1 said the paper's pipeline would score F1 < 0.5; it scored 0.56 (and zero-fill 0.69), so **H1 is not met
by its number** — my peak detector is lenient. The failure is plain in the images and in the energy-on-target
column, but the line I set was the wrong line, and it stays as set.

![isar phase transition](figures/isar_phase_transition.png)

The phase transition of the image-domain pipeline on random scenes: 80 scatterers need ≥ 10 % of the
samples, 160 need ~20 % and are unreliable, 320+ fail at every rate tried (FISTA, 200 iterations, fixed λ —
the pipeline's transition, not the information-theoretic one). The paper's target sits well inside the
feasible region at 22.5 %.

![isar resolution](figures/isar_resolution.png)

Two equal scatterers on a 2× oversampled image grid with only the in-band k-samples measured (so
resolving below one cell is a real question, not a grid artefact): ℓ₁ resolves them **0.6 Fourier cells
apart** at every SNR from 10 dB, and not at 0.4. On-grid, noise-only — the classic sparse super-resolution
result, reproduced.

![isar motion](figures/isar_motion.png)

A 3-cell translational range walk over the CPI smears the full-data image to F1 0.03. Compensating the
walk on the *kept* samples and then running image-domain ℓ₁ gives F1 1.00 — the step that was "a disaster"
on the paper's compressive image behaves normally once the image, not the data, is the sparse unknown.

## 2. The master's thesis, reproduced honestly

![thesis example 1](figures/thesis_example1.png)

The thesis's test signals (§4.1) are five tones with amplitudes 10…10⁵ and frequencies aᵢ·rᵢ^{rᵢ} — up to
10¹⁴ rad/s on 400 samples over [−1, 1]. Aliased that many times over, the discrete signal is five tones at
off-bin frequencies, and its ten largest DFT coefficients hold **93.4 %** of its energy, not 100 %. It is
not sparse on the grid, so:

| on the thesis's Example 1 | relative error |
|---|---|
| FTSA (keep the K = 6 largest Fourier coefficients; *sees the whole signal*) | 0.28 |
| ℓ₁ from 70 non-uniform samples (interior point, the thesis's `l1eq_pd` setting) | 0.36 |

The thesis reports 2.5e-5 for the ℓ₁ step and notes that "in order for the algorithm to work it needs to
have the original signal" (p. 41). That is the tell: the recovery was never from the samples alone.

![thesis sampling](figures/thesis_sampling.png)

With tones that are actually sparse on the grid, the thesis's own sampler and an interior-point ℓ₁ solve
do exactly what the thesis hoped: relative error ~1e-7 in **100 % of draws down to 13 % of the samples**,
and 95 % of draws at 9 %. The thesis's 10⁴ amplitude range does not trouble ℓ₁ at all — but it breaks FTSA,
whose fixed-ratio threshold keeps 2 of the 10 coefficients (error 0.10). Pre-registered claim T1 (≥ 90 % of
draws below 1e-3 at 17 %) **holds for sparse signals and fails for the thesis's own recipe**.

Also as in the thesis: chirp trains are not Fourier-sparse (FTSA 0.61, ℓ₁ 1.0 — both fail); the
Walsh–Hadamard variant is far worse than the DFT for tones (0.31 with 116 coefficients vs 1e-14 with 10).

![thesis radial](figures/thesis_radial.png)

The thesis's last experiment (§6.2, the Candès–Romberg–Tao phantom from radial Fourier lines, run through
`l1-magic`) with our own TV solver: 18 lines → 5 % error, **22 lines → 2.4 %**, 40 lines → 0.8 %, against
51 % / 37 % for zero-filling. The exact Fourier-domain x-update
([`tv_fourier`](csi/solvers.py)) is what makes this converge; a generic conjugate-gradient inner solve did not.

## 1. Solvers, from scratch

[`csi/solvers.py`](csi/solvers.py) — each with a test in [`tests/test_solvers.py`](tests/test_solvers.py)
against a property it must have:

| solver | tested against |
|---|---|
| FISTA | Beck–Teboulle's bound F(xₖ) − F* ≤ 2L‖x₀ − x*‖²/(k+1)², every iteration |
| ISTA | monotone descent; ~1 % behind FISTA's objective after 6,000 iterations |
| OMP | exact recovery of an 8-sparse vector from 64 Gaussian measurements of 256 |
| CoSaMP | exact recovery; error proportional to noise |
| IRLS, ADMM basis pursuit, interior point | agree with each other to 1e-4; interior point's duality gap < 1e-8 |
| ADMM-TV | denoises a piecewise-constant image below the noise level |

The interior-point method (primal log barrier, Newton with the KKT system reduced by elimination, Boyd &
Vandenberghe ch. 10–11) is the algorithm behind `l1-magic`'s `l1eq_pd` — the thing the thesis said it wanted
to learn. It converges in ~70 Newton steps on the thesis-sized problems.

## 4. The video paper, reproduced and extended — mostly a negative result

The 2015 paper's setting: the video is degraded by Gaussian noise (σ = 15/255), each column of each frame is
measured by a Gaussian matrix with M of N rows (the paper: 230 of 256), the column is assumed sparse in the
DCT, and FISTA ("FGbCS") is compared with OMP, CoSaMP, IRLS on PSNR against the clean frame. Reproduced on
12 Sintel frames (128 × 256) since the paper's *salesman* clip is no longer hosted.

![video](figures/video.png)

| M/N = 0.9, identical measurements | PSNR |
|---|---|
| **FISTA** (the paper's method) | **24.6 dB** |
| IRLS | 24.2 |
| CoSaMP | 23.2 |
| OMP | 22.5 |
| the noisy input, untouched | 24.7 |

**The paper's ranking reproduces** — FISTA > IRLS > CoSaMP > OMP, exactly its order. **Its gain does
not**: on a textured frame, column-wise DCT sparsity is too weak a prior to denoise with. The best λ gives
+0.5 dB over the noisy input at M/N = 0.9 and is 2 dB *below* it at M/N = 0.5 (λ swept 0.002–0.2; 0.02 is the
optimum and is what the curves use). The paper's ~4 dB gain on *salesman* — a smooth, low-texture clip —
does not transfer to this footage.

**Time and motion.** Recovering frame *differences* against the previous reconstruction is worse than
independent recovery at every rate (drift plus a residual that is no sparser in a column DCT). Predicting
the frame along the flow from `optical-flow-inverse` and recovering only the residual is the best of the
three, by **0.3–0.5 dB** from M/N = 0.4 upward. Pre-registered V1 (30 dB with ≥ 30 % fewer measurements)
is **not supported**: no scheme reaches 30 dB at any rate tried, because the model cannot denoise this far.
The honest summary is that motion helps a little and the sparsity model is the bottleneck.

## Run it

```bash
pip install -e ".[dev]"
pytest                          # 20 tests, ~7 s
python scripts/run_isar.py      # Part 3, ~1 min
python scripts/run_thesis.py    # Part 2, ~1 min
python scripts/run_video.py     # Part 4, ~1 min; needs optical-flow-inverse installed and its Sintel frames
```

MIT license.
