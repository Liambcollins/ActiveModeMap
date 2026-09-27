<!-- 2026-08-14. Paths updated for the tidied layout; see ../MIGRATION.md. -->

# Complex-field reconstruction (amplitude AND phase), bias recovery, uncertainty

Deck: `reporting/ActiveModeMap_spectra_bias_uncertainty.pptx`. Built standalone,
NOT on `ORNL template 1.pptm` — that file is in neither connected folder, so the
theme colours are set literally from `deck-physics-informed-al.md`.

## Correction to a claim made earlier the same day

A first version of this deck said the EB arm could not be rebuilt because
`eb_models` and `geometry` were missing from `EB-Solver-CResonance/src`. **That
was wrong.** They are in `EB-Solver-CResonance/fem_eb_comparison/py/`; `src/`
holds only the `eb_cr_afm` package, which is why looking there alone was
misleading. Both modules import fine and `solve_single_frequency` returns
complex `z`, so the EB library rebuilds in 6.3 min and the EB arm is in the
deck. Recorded here because the same mistake is easy to repeat: **the EB
wrapper lives in `fem_eb_comparison/py`, not in `src`.** `config.add_eb_to_path()`
now puts both directories on `sys.path`.

Separately, the EB *results* were never missing — `results/phys_bench.csv`
(540 rows: `eb`, `eb_gp`, `lowrank`), `ebic_bench.csv` (648 rows, k₂=k₁ +
compliant cone) and `eb_f2f1.npy` were all in the repo and should have gone
on the amplitude slide regardless of the library question.

## The one change that made phase possible

Both published library builders discard the phase —
`archive/fem/build_fem_library.py` at `np.abs(...)`, `libraries/buildlib.py` by
storing `log|z|`. Two new builders keep it:

- `archive/fem/build_femlib_cplx.py` — re-stitches the 160 isotropic ladder
  exports keeping `re_u`/`im_u` beside an unchanged `logamp_u`. All five
  original checks pass (32/32 rungs, f_res monotonic 312.4→375.9 kHz, Q
  strictly decreasing with damping on every rung, Q range 55–472 brackets the
  measured 191, 0.00 % NaN). 26 s.
- `libraries/buildlib_cplx.py` — 96 k₁ × 4 g × 1600 frequencies through the EB
  solver, written pre-aligned. f_res monotonic 339.2→383.0 kHz, 0.00 % NaN.
  6.3 min.

**Two interpolation conventions differ between the EB and FEM paths and getting
them backwards costs several percent.** Position: `buildlib.py` interpolates
|z| over x then logs. Frequency: physrec's raw path interpolates the *log*
amplitude linearly in f, while `build_fem_library.py` interpolates the
amplitude and logs after. Both are reproduced verbatim, which is why:

| arm | published | rebuilt |
|---|---:|---:|
| EB | 6.53 % | 6.53 % |
| EB + GP | 2.41 % | 2.41 % |
| FEM | 6.31 % | 6.31 % |
| FEM + GP | 1.94 % | 1.94 % |
| GP only | 3.41 % | 3.41 % |

k₁, f_free and g match to the digit as well (EB 374.0 N/m, FEM 1152.8 N/m).

## The complex arms

`src/cplxrec.py`. Physics parameters come from the **exact published objective**
(`physrec.fit_eb` on log(A+F) over the revealed SNR window); only the forward
*prediction* changes, taken from the complex channel and scaled by one complex
gain.

**The gain estimator matters more than it looks.** The complex least-squares
gain ⟨M,Z⟩/⟨M,M⟩ put the arm at 100 % error *with the amplitude wrong* — the
measured phase runs 180° across the resonance, so residual model–data phase
error makes that coherent sum cancel and |γ| came out ~10× small. Splitting it
— |γ| from the published mean-of-log-ratios, arg γ from the amplitude-weighted
**circular** mean of arg(Z/M) — fixes it and the amplitude then matches the
published arm to 0.03 %.

## Headline: both forward models get the amplitude right and the phase wrong

Held-out, n = 8 equispaced (truth D-NS 224.09 µm):

| arm | amp NRMSE | phase MAE | complex NRMSE | D-NS err | k₁ (N/m) |
|---|---:|---:|---:|---:|---:|
| GP only | 2.87 % | 0.7° | 3.5 % | 4.66 µm (no crossing) | — |
| **EB alone** | 6.44 % | **42.4°** | **142 %** | 1.82 µm | 374 |
| EB + GP | 2.93 % | 0.8° | 3.8 % | **0.15 µm** | 374 |
| **FEM alone** | 6.28 % | **41.4°** | **144 %** | 1.00 µm | 1153 |
| FEM + GP | 2.94 % | 0.8° | 3.8 % | **0.15 µm** | 1153 |

**This is the new result.** Two unrelated forward models — a two-segment beam
and a geometry-faithful beam-network FEM, fitted to contact stiffnesses 3×
apart — fail by the same 41–42° and take the antiresonance branch the same
wrong way. That rules out a geometry-specific quirk and points at the drive or
detection model. It also is not a mis-set resonance: `cplxrec.fit_cplx` refits
(f_res, g) against the complex residual on the revealed block, moves f_res
292.80 → 295.15 kHz, drives damping to the library edge and makes amplitude
*worse* (59.5 %). That function is kept, and documented as a failure.

Phase is nonetheless the sensitive channel for (f_res, Q) — 0.1 % of f_res is a
fifth of the half-width on a Q ≈ 200 line — so a complex-domain fit is worth
doing once a model can represent the phase.

Node accuracy is where the physics pays: both +GP arms are 0.15 µm at every
n ≥ 5, while GP-only never finds a crossing in span at any budget from 4 to 30
positions (≈3 µm out every time).

## Bias dependence, re-derived and tied to the mode map

`src/bias.py`, from `series_checkpoint.npz` (7 biases × 7 positions × 16 000
bins), resolved through `config.BIAS_SERIES`. Peak search **banded to
285–302 kHz**: the wide band let the finder hop to mode B at the bias/position
where mode A collapses (a 44 kHz jump at −2 V, x = 224). With the tight band
every position's peak moves < 1 kHz across ±6 V.

- |Z| at mode A collapses ~100× at V ≈ −1.9 V; Z(V) is a straight line through
  the origin to 3.5–4.4 %.
- **V_cpd = −1.86 ± 0.08 V** over the 5 mode-A-sensitive positions (224/225 sit
  on the node and cannot constrain it). An earlier note said −1.90 ± 0.08 from a
  band-weighted estimator — same answer.
- Q(V = 0) = 206 ± 4 by interpolated FWHM at each spectrum's own peak.
- |b| spans **43×** along the lever, |P⊥| only **7×**, corr(log|P⊥|, log|b|) =
  **−0.38**. Residual = **0.077 ± 0.082 V** equivalent DC offset = 1.3 % of a
  6 V drive. No measurable piezoresponse.

**New:** the sparse mode map *predicts* the bias sensitivity. Both EB + GP and
FEM + GP match the measured electrostatic slope |∂Z/∂V| to ≤ 4 % over
132–196 µm with a single free scale (3 overlapping sensitive positions, 2 dof).
At 224/225 µm even the dense 1500 nN map disagrees by 44–58 %, because that
series ran at 1000 nN and near a null a small contact-stiffness change is a
factor, not a percent — a caveat, and a reminder that node-adjacent
measurements are exquisitely load-sensitive.

## Uncertainty

σ is the predictive sd of one real component of Z, split into a parameter term
(12-fold LOO refit of k₁, f_res, g) and a discrepancy-GP term (per-frequency
variance, shared PRESS-selected length scale). Leakage-safe throughout.

- **σ/|Ẑ| ≈ 0 dB on the antiresonance trough and at the node** — the map is
  least trustworthy exactly where the features are.
- The GP term carries **~4×** more of σ than the parameter term at n = 8.
- Calibration (rms standardised residual z, 1.00 = honest):

| n | 5 | 6 | 8 | 10 | 12 | 16 | 20 | 30 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FEM + GP | 0.59 | 0.67 | 0.82 | 0.92 | **0.99** | 1.17 | 1.20 | 1.63 |
| EB + GP | 0.48 | 0.55 | 0.67 | 0.72 | 0.77 | 0.81 | **1.05** | 1.42 |
| GP only | 0.39 | 0.42 | 0.49 | 0.50 | 0.55 | 0.45 | 0.48 | 0.56 |

  Both physics arms start cautious and end **over-confident** — σ shrinks faster
  than the error, a real defect worth fixing. GP-only is ~2× over-cautious at
  every budget.
- **At n = 4 no discrepancy GP is fitted at all** and σ is the parameter term
  alone: wrong by 16× (EB) and 47× (FEM). Never quote an error bar from four
  positions.

## Still out of scope

Mode B, the nominal-axis control, non-equispaced designs, and the `eb_ic`
variant (k₂ = k₁ + compliant cone, `archive/eb_variants/buildlib_ic.py`) — the
last would be the natural next arm, since the Q6 EB/FEM confound is that EB has
k₂ = 0 while the FEM arm is isotropic.

## Reproduce

```
python archive/fem/build_femlib_cplx.py   # 26 s from the 160 isotropic .h5
python libraries/buildlib_cplx.py         # 6.3 min through the EB solver
python src/run_recon_cplx.py              # n = 8, six arms -> results/recs_cplx.pkl
python src/sweep_n_cplx.py                # n = 4 ... 30 -> results/sweep_n.pkl
python src/bias.py                        # -> results/bias_analysis.npz
python figures/d_cplx_spectra.py figures/d_bias_recovery.py figures/d_uncertainty.py
python reporting/build_deck_cplx.py
```

`physrec.py` needed one edit: its module-level `np.load(OUT/'eblib.npz')` is now
guarded, so the module imports on a machine that has only one of the two
libraries. `use_library` already rebinds every global it touches. Both
libraries must be loaded and released in turn — each carries 0.5–0.8 GB of
Re/Im blocks, and a half-swapped physrec would silently mix forward models.
