# Sparse contact-resonance mode mapping: two-segment EB, EB+GP, and GP alone

Retrospective, leakage-safe replay on the dense mode-shape reference
`Dense_Grid_A` (101 positions × 16 000 frequency bins, Multi75G on SCM-PIT).
The question: **from how few measured positions can we reconstruct the full
complex response along the lever, and does a physical forward model earn its
keep against a well-tuned Gaussian process?**

Three arms carry every headline number here:

| arm | what it is | where |
|---|---|---|
| **EB (2-segment)** | analytic Euler–Bernoulli beam, contact spring `k₁`, `k₂ = 0`, φ = 11°, 15.76 µm overhang — precomputed into a resonance-aligned surrogate library so each fit is pure interpolation | `libraries/buildlib*.py`, `src/physrec.py` |
| **EB + GP** | the same EB fit plus an isotropic squared-exponential **discrepancy** GP over position, one length scale shared across frequency, chosen by closed-form LOO PRESS on the revealed block | `src/physrec.py`, `src/cplxrec.py` |
| **GP alone** | no forward model — the same kernel machinery with a per-frequency constant mean. The honest baseline | `src/run_gp.py`, `cplxrec.rec_gp_cplx` |

Everything else — the geometry-faithful FEM arm, the low-rank baseline, the
compliant-cone EB variant, the AFeMulator ladder driver — is in `archive/`,
still runnable, with `archive/README.md` explaining why each was demoted.

## Headline results

- **Amplitude, band A, n = 8 revealed positions:** GP alone 3.41 % NRMSE,
  EB+GP 2.41 %, FEM+GP 1.94 %. Bare EB 6.53 %, bare FEM 6.31 %.
- **The detection node** (the 2-decade amplitude null at x = 224.09 µm):
  EB+GP and FEM+GP both land 0.15 µm away. **GP alone never finds a crossing
  in span, at any budget** — this is the one place the physics is not a
  refinement but the difference between finding the feature and not.
- **Phase fails structurally, in both forward models.** EB 42.4° mean absolute
  phase error, FEM 41.4°; complex NRMSE 142 % / 144 %. At contact stiffnesses
  a factor 3 apart, the two models fail *identically*, so the missing physics
  is in the drive or detection model, not the geometry.
- **k₁ is not identifiable from band A alone.** The mode ratio is what pins it:
  measured f₂/f₁ = 3.0794, which EB reaches at **k₁ ≈ 1044 N/m**. Two-band
  fitting with a per-band gain returns **k₁ = 989 ± 15 N/m across n = 3–30**,
  and places the mode-2 peak inside one 437 Hz bin. (FEM-isotropic saturates at
  f₂/f₁ = 2.942 — no k₁ gets it to the measured ratio.)
- **Per-band gain ratio −9.51 ± 1.24 dB** = mode 2 driven 2.99× more weakly
  than mode 1, against 3.11× measured independently from the domain-flip data.
- **Wide band (25 kHz–1.775 MHz):** the physics libraries cover 25 % of the
  span (band-A) or 49 % (two-band); the GP covers 100 %. Where both are
  defined the medians tie within 0.06 dB, but physics wins the tail —
  99.1–99.7 % of signal bins within 3 dB vs 96.6–98.0 %. Note ~85 % of the
  measured span sits at the noise floor.
- **GP alone plateaus with n:** median error 0.20 → 0.12 dB from n = 5 to 30,
  but the fraction of signal bins within 3 dB barely moves, 97.4 → 97.8 %.
- **Acquisition chose nothing.** Every number in this study and in the deck
  uses **equispaced** positions. Every adaptive strategy tried was worse
  (GP-only max-variance 12.94 % vs 3.41 % equispaced at n = 8; 44.20 % vs
  6.82 % at n = 5). See `docs/` and `figures/d_acq.py`.

## Layout

```
config.py              every path, overridable with AMM_* env vars
src/
  physrec.py           EB/FEM surrogate fitting, discrepancy GP, GP-only, low-rank
  cplxrec.py           the same arms on the COMPLEX field, plus a predictive sigma
  twoband.py           joint mode-1 + mode-2 fit, per-band gain
  bias.py              bias-dependence analysis (V_cpd, the piezoresponse bound)
  setup_data.py        loads Dense_Grid_A, applies the InvOLS position calibration
  harness.py           the leakage-safe reveal loop and scorer
  run_phys.py          EB / EB+GP benchmark             -> results/phys_bench.csv
  run_gp.py            GP-only baseline (+ oracle-ell)  -> results/gp_bench.csv
  run_recon_cplx.py    complex reconstructions at n = 8 -> results/recs_cplx.pkl
  sweep_n_cplx.py      sigma and true error vs n        -> results/sweep_n.pkl
  run_twoband.py       dual-peak fit, 4 arms x 10 budgets -> results/dualpeak.csv
  run_wideband.py      full measured band, all arms     -> results/wideband.csv
libraries/
  buildlib.py          EB k2=0, log-amplitude only (the published library)
  buildlib_cplx.py     EB k2=0 WITH Re/Im, 185-720 kHz         (~6 min)
  buildlib_x_cplx.py   EB extended to 1350 kHz for two-band    (~13 min, 870 MB)
figures/               one script per deck figure (ORNL palette in ornl.py)
reporting/             build_deck_cplx.py -> the 12-slide standalone deck
results/               CSVs, small derived arrays, figure PNGs (libraries are not committed)
docs/                  findings write-ups, including every correction made
notebooks/             the reproduction path, in order
archive/               demoted arms, still runnable — see archive/README.md
```

## Reproduce

```bash
export AMM_DENSE_GRID_A=/path/to/SCM_PIT/Dense_Grid_A
export AMM_EB_REPO=/path/to/EB-Solver-CResonance

python libraries/buildlib_cplx.py        # band-A complex library, ~6 min
python libraries/buildlib_x_cplx.py      # extended library for two-band, ~13 min

python src/run_phys.py                   # EB, EB+GP
python src/run_gp.py                     # GP alone
python src/run_recon_cplx.py             # complex maps at n = 8
python src/sweep_n_cplx.py               # sigma calibration vs n
python src/run_twoband.py                # dual-peak fit
python src/run_wideband.py               # full measured band

python figures/d_cplx_spectra.py figures/d_uncertainty.py \
       figures/d_twoband.py figures/d_wideband.py figures/d_tf_shapes.py
python reporting/build_deck_cplx.py
```

Every script finds `config.py` by walking up from its own location, so it runs
from any working directory and from `archive/` one level deeper.

External dependencies are *not* vendored:

| dependency | used for | pointed at by |
|---|---|---|
| `EB-Solver-CResonance` | the two-segment EB forward model (`eb_cr_afm`, `eb_models`, `geometry`) | `config.EB_REPO` |
| `activemodemap` (this repo) | D-NS classifier, position calibration, low-rank baseline | normal install |
| AFeMulator (GUI, HTTP on `127.0.0.1:8050`) | the FEM arm only | `archive/fem/run_fem_ladder.py` |

Note `eb_models.py` and `geometry.py` are in `EB-Solver-CResonance/fem_eb_comparison/py/`,
**not** in that repo's `src/`. `config.add_eb_to_path()` now puts both on
`sys.path`; before this tidy it added only `src/`, so `libraries/buildlib.py`
worked only when run from inside `fem_eb_comparison/py`.

## Ground rules that make the numbers defensible

- **Leakage-safe.** A reconstructor's signature is
  `(x_grid, sel_idx, Z_sel, freq) -> dict(Zrec, std)`. The noise floor, the SNR
  window, the complex gains, the GP length scale, the per-frequency variance and
  the low-rank rank are all re-estimated from the revealed block at every step.
- **The scorer alone** sees withheld positions.
- **Baselines must be strong.** The GP-only arm exists because the low-rank
  baseline flattered the physics (`docs/gp-only-baseline.md`).
- **Corrections stay in the record.** `docs/` keeps every claim that turned out
  to be wrong alongside what replaced it — the EB "3.4× too soft" reading, the
  wide-band k₁ blow-up, the two-band local minimum at k₁ ≈ 500 N/m.

## Known caveats

- The complex gain must be estimated as |γ| from mean-of-log-ratios and arg γ
  from an amplitude-weighted **circular** mean. A complex least-squares gain
  ⟨M,Z⟩/⟨M,M⟩ cancels — measured phase swings 180° through resonance — and
  returns |γ| about 10× too small (100 % error with the amplitude wrong).
- `cplxrec.fit_cplx` (refining `f_res`, `g` on the complex residual) is kept but
  **documented as a failure**: it drives damping to the library edge.
- The two-band coordinate refinement has a strictly worse local basin at
  k₁ ≈ 500 N/m (cost 0.607 vs 0.353 at k₁ = 1008). `coarse=(48, 15, 7)` is
  required; the published run's stability at small n came from its 16-point grid
  happening to straddle the good basin, not from robustness.
- `strat_maxvar` selects on the *low-rank* reconstructor's variance no matter
  which arm is being scored. The per-component σ from `cplxrec` has never been
  fed back into acquisition — that is the obvious next experiment.
- At n = 4 no GP is fitted, so the reported σ is wrong by 16×/47×; `+GP` arms
  below n = 5 ARE the bare-physics arms.
