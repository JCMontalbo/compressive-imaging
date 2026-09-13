# compressive-imaging — pre-registered plan

*Written before any experiment was run. Results are reported against it either way.*

## What this repo is

My master's work (UTRGV, 2016) and two SPIE papers looked at sparse approximation and ℓ₁ recovery for
radar signals and noisy video, leaning on off-the-shelf solvers (`l1-magic`) and toy signals. The ISAR
paper's reconstruction did not work, and said so. The thesis's conclusion named the next step:
*"learn more about interior point algorithms so as to try and develop our own compressive sensing
algorithm that can be utilized within the radar system process."*

This repo does that next step: solvers written from scratch and tested against theory; the thesis's
experiments reproduced with honest labels; the 2016 ISAR model rebuilt and the failure diagnosed; the
2015 video method reproduced and extended with motion.

Sources: J. Montalbo, *Compressive Sensing and Radar Imaging*, MS thesis, UTRGV 2016 (ch. 4–6);
Hu, Montalbo, Li, Sun, Qiao, *Sparse representation for the ISAR image reconstruction*, Proc. SPIE 9857
(2016); Zhao, Montalbo, Li, Sun, Qiao, *Compressive sensing for noisy video reconstruction*, Proc. SPIE
9484 (2015).

## Part 1 — solvers (`csi/solvers.py`)

ISTA, FISTA (Beck–Teboulle; the 2015 paper's "FGbCS" is FISTA), OMP, CoSaMP, IRLS, ADMM for
TV-regularised least squares, and a primal log-barrier interior-point method for basis pursuit
(`min ‖x‖₁ s.t. Ax = y`, the problem `l1-magic`'s `l1eq_pd` solved). Complex-valued where the
measurement is Fourier. Every solver has a test against something it must satisfy:

- FISTA: objective gap ≤ 2L‖x₀−x*‖²/(k+1)² (Beck–Teboulle Thm 4.4) on a problem with known x*.
- ISTA vs FISTA: same fixed point; FISTA reaches 1e-6 in fewer iterations.
- OMP: exact recovery of a k-sparse vector from a Gaussian matrix when m ≥ 4k log n (with margin).
- CoSaMP: same, and robust to noise (error ∝ noise level).
- IRLS: converges to the ℓ₁ solution of basis pursuit within 1e-4 of the interior-point answer.
- Interior point: KKT residuals below tolerance; agrees with ADMM basis pursuit to 1e-6.
- ADMM-TV: piecewise-constant signal recovered from noisy measurements with error below the noise.

## Part 2 — the master's thesis, honestly (`scripts/run_thesis.py`)

The thesis's signals: five-tone trains `Σ aᵢ sin(cᵢ t)` with the amplitude/frequency recipe of §4.1, and
five-chirp trains of §4.5; 400 samples; the thesis's random-step non-uniform sampler.

- **FTSA** (keep the K largest Fourier coefficients of the *full* signal) reproduced as the thesis
  describes it and labelled what it is: an oracle transform-coding bound, not compressive sensing —
  the thesis itself notes it "needs to have the original signal" (p. 41).
- **ℓ₁ from non-uniform time samples** (the thesis's ch. 6 experiment, which used `l1-magic`) with our
  interior-point and FISTA solvers, DFT dictionary. The thesis reports 67 of 400 samples (16.75 %),
  error ≈ 2.5e-5; that becomes one point on a sampling-fraction vs. error curve over 20 random draws.
- **Walsh–Hadamard** variant (§4.3) as the thesis did, for completeness.
- **TV from radial Fourier lines** (§6.2, the Candès–Romberg–Tao phantom experiment) with our ADMM-TV:
  reconstruction error vs. number of lines, the thesis's 40 lines marked.

Pre-registered claim T1: for the five-tone signals (10-sparse in Fourier), ℓ₁ from 17 % non-uniform
samples recovers the signal to relative error < 1e-3 in ≥ 90 % of random draws; FTSA at the same
budget of *coefficients* (10) does better only because it sees the whole signal — reported side by side.

## Part 3 — the ISAR paper, rebuilt and diagnosed (`csi/isar.py`, `scripts/run_isar.py`)

The paper's model, built properly: point-scatterer target (its pseudo-aircraft, ~30 scatterers on a
cross), turntable ISAR, stepped-frequency 3.0–3.384 GHz (384 MHz bandwidth), pulses at 20 kHz over the
coherent processing interval at 0.15 rad/s, far-field Born phase history
`D(f, θ) = Σᵢ σᵢ exp(−j4πf rᵢ(θ)/c)`, polar→rectangular interpolation, image by 2D FFT. Compressive
sampling as in the paper: keep a uniformly random 22.5 % of the samples (45 % of Nyquist).

Two pipelines on the same samples:

- **P (the paper's)**: find the sparsest representation of the *data* `z` with `y = Fz` (partial Fourier
  of the phase history), then form the image from the recovered data. Sec. 3 steps (b)–(c).
- **I (image domain)**: treat the kept samples as partial Fourier measurements of the *image* and solve
  `min ‖x‖₁ s.t. ‖PFx − y‖ ≤ ε` for the image directly.

Pre-registered:

- **H1 (the failure is real)**: pipeline P at 22.5 % sampling gives an image whose scatterer-detection
  F1 (peaks within one resolution cell of true scatterers) is < 0.5 — reproducing Figure 4's smudge.
- **H2 (the diagnosis)**: pipeline I at the same 22.5 % samples gives F1 ≥ 0.9 and localisation error
  < 1 cell for ≥ 90 % of scatterers, at the paper's SNR (noise-free) and at 20 dB.
- **H3 (why)**: the phase history's best 30-term Fourier approximation captures < 50 % of its energy
  (it is not sparse in the basis P assumes), while the image is exactly 30-sparse.
- Then: a phase transition (sampling fraction × scatterer count → P(F1 ≥ 0.9)), and a two-scatterer
  resolution test — separation vs. SNR at which ℓ₁ still resolves them below the Fourier limit.
- Motion compensation (the step that "was a disaster" on the paper's CS image) applied to the pipeline-I
  image: it should now behave as it did on the fully-sampled image.

If H2 fails — if image-domain ℓ₁ also fails at 22.5 % — the honest conclusion is that the sampling rate,
not the representation, was the problem, and the phase transition will say where it stops failing.

## Part 4 — the video paper, reproduced and extended (`scripts/run_video.py`)

Frames of the Sintel shot already used in `optical-flow-inverse` (the paper's *salesman* clip is no
longer hosted), 256 px, Gaussian noise σ = 15/255, per-column random Gaussian measurements with M = 230
of N = 256 as in the paper, DCT sparsifier. FISTA (= FGbCS, λ = 20/255² scaled, K = 40) against our own
OMP, CoSaMP, IRLS on identical measurements. PSNR vs M curve as in the paper's Fig. 5.

Then time:

- **D (differences)**: measure and recover frame differences `fₖ − fₖ₋₁`, which are sparser.
- **M (motion-compensated)**: predict `f̂ₖ` from `fₖ₋₁` along the flow recovered by `optical-flow-inverse`
  and recover only the residual `fₖ − f̂ₖ`.

Pre-registered V1: at 30 dB, M needs ≥ 30 % fewer measurements per frame than frame-independent
recovery; D sits between. If M ≈ D, the flow adds nothing over plain differencing on this footage.

## Discipline

Every figure and number from a script; a test per claim; CI on 3.10–3.13; misses reported beside hits.
