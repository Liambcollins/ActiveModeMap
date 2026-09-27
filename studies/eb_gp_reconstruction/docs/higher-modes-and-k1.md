# The off-resonance mismatch is a mode-2 problem — and mode 2 is where k₁ lives

## The measured higher mode

Full sweep, 25–1775 kHz, max over 101 positions:

| f (kHz) | level | identity |
|---|---|---|
| 292.97 | 0 dB | flexural contact mode 1 |
| 902.41 | −4.9 dB | flexural contact mode 2 |
| 970.55 | −25.8 dB | probably lateral (21× weaker) |

**Measured f₂/f₁ = 3.0802** (clamped–free 6.267, clamped–pinned 3.240,
clamped–clamped 2.757 — the contact behaves close to a pin). The 902 kHz
assignment is safe: in the FEM exports lateral modes are 60–80× weaker than
flexural in `z_displacement`.

## Every benchmarked model puts mode 2 too low

| model | k₁ | f₂/f₁ |
|---|---|---|
| **measured** | — | **3.080** |
| EB 2-seg (k₂=0) | 346 (band-A fit) | 2.484 |
| EB 2-seg (k₂=0) | **1046** | **3.080** ← matches |
| EB 2-seg (k₂=0) | 20000 | 3.213 (saturates) |
| FEM frictionless | 5000 | 2.660 (saturates) |
| FEM isotropic | 5000 | 2.942 (still rising) |

EB reaches the measured ratio at k₁ ≈ 1046 N/m — consistent with Hertz (1310)
and the mode-A shape-fit range (1000–2154), and 3× stiffer than the band-A
reveal-loop fit. FEM cannot reach it in either lateral condition within the
swept range (`run_fem_etip.py` tests whether the elastic tip column is why).

## The observable consequence: the antiresonance

The band-A antiresonance lives between modes 1 and 2, so a low mode 2 drags it
down. Fitted models vs data over x > 200 µm: EB −18.2 kHz (5.3 %), FEM
−11.2 kHz (4.1 %) — both low, in the same order as their f₂/f₁ error. This is
the mechanism behind the ~8 dB-high off-resonance floor.

## What including the full frequency range does

**Naively widening the fit window is catastrophic** (n = 20): EB k₁ → 150 N/m
(library edge), NRMSE 5.39 → 34.74 %; FEM 1015 → 223 N/m, 6.98 → 28.65 %.
380 off-resonance bins outvote 32 resonance bins to chase a floor the model
cannot represent.

**Including the mode-2 resonance would fix the biggest weakness.** Over
k₁ = 328 → 1291 N/m, EB's f₁ moves +3.8 % while f₂/f₁ moves +28.0 % — the
ratio is **7.3× more sensitive to k₁**, and f₁ saturates above ~700 N/m while
the ratio keeps climbing. Band A carries the mode shape (and the D-NS); band B
carries the stiffness. Fitting only band A gets a good map from a badly wrong
k₁.

## The corrected Q6 statement

> Neither model's k₁ is identified by the mode-1 band. FEM's fitted value lands
> inside the independent estimates and EB's does not, but the mode-2 ratio
> shows EB reaches the right stiffness at k₁ ≈ 1046 N/m — an
> **identifiability** result, not evidence that one model's parameters are more
> physical. Adding band B is what settles it.

## Two-band fitting: the next experiment, and it is cheap

The data exists (all positions swept to 1775 kHz; `setup_data.BANDS['B'] =
(860, 1300) kHz`). Needed:

1. Extend both libraries past mode 2 — aligned frame U_HI ≈ 3.5
   (`build_fem_library.py`), EB `FLIB` to ~1.35 MHz (`buildlib*.py`). The FEM
   raw exports already contain 0–2 MHz; no new FEM runs.
2. Fit both bands jointly with a separate SNR window per band.
3. Prediction: k₁ → ~1000–1300 N/m, antiresonance error drops; band-A NRMSE may
   worsen slightly while the physics gets more correct — report either way.

## Reproduce

`figures/d_modes.py`; EB curve in `results/eb_f2f1.npy` (26 k₁ values ×
900-point sweeps of `eb_cr_afm.solve_single_frequency`); FEM ratios from
`eigen/frequencies_hz` filtered to `mode_types == 1`.
