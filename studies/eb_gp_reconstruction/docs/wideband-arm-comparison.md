<!-- 2026-08-14. Paths updated for the tidied layout; see ../MIGRATION.md. -->

# Which arm wins across the FULL measured band — 25 kHz to 1.775 MHz

Answers "which does best across the full measured frequency space, as well as GP
alone > 5". Supersedes nothing; it extends `wideband-accuracy-verdict.md`
(which scored band A in dB) to the whole measurement and adds the EB arms.

Figure `results/fig/d6_wideband.png`, every number in `results/wideband.csv`.
Generators: `src/run_wideband.py`, `figures/d_wideband.py`,
`libraries/buildlib_x_cplx.py`.

## The finding that reframes the question: the arms do not share a domain

| arm | library u range | frequency reach | % of the measured band |
|---|---|---|---:|
| GP only | — | every measured bin | **100 %** |
| EB + GP, FEM + GP | 0.55–2.05 | 161–600 kHz | **25 %** |
| EB + GP, two-band | 0.55–3.45 | 161–1011 kHz | **49 %** |

Mode 2 sits at u = f/f_res = 3.08, outside the band-A libraries entirely.
**`predict_cplx` and `eb_predict` both CLIP u to the library range**, so asking a
band-A arm for 902 kHz returns the library edge — smooth, plausible, and
meaningless. `cplxrec.coverage(freq, f_res)` now exists for exactly this, and
every score below is masked with it.

A second trap, found the same way: handing `rec_phys_cplx` the whole 25–1775 kHz
axis lets `physrec.snr_window` include mode 2 (it clears the 10 %-of-max
threshold at 57 % of mode 1), so a band-A library gets asked to fit a resonance
it cannot represent and **fitted k₁ wanders from 111 to 2710 N/m**. Restricting
the fit to 265–500 kHz puts k₁ back on the published value (EB 388, FEM 1254 at
n = 8) — hence the new `fit_band` argument. Score wide, fit where the library
lives.

## Median |error| in dB, held-out positions, n = 8

Regions are fixed frequency edges, not data-driven, so they mean the same thing
for every arm. "—" = no library coverage.

| region | GP only | EB + GP | FEM + GP | EB + GP (2-band) |
|---|---:|---:|---:|---:|
| below mode 1 (25–265 kHz) | 0.17 | — | — | — |
| mode 1 ridge (285–302) | 0.12 | **0.10** | **0.10** | 0.11 |
| mode 1 antiresonance (302–500) | 0.31 | 0.30 | **0.28** | 0.32 |
| valley, modes 1–2 (500–700) | **0.40** | 0.53 | 0.44 | 0.52 |
| mode 2 flank (700–880) | 0.25 | — | — | **0.22** |
| mode 2 ridge (880–925) | 0.30 | — | — | **0.18** |
| above mode 2 (925–1775) | 0.37 | — | — | — |
| **signal bins within 3 dB** | 97.6 % | **99.5 %** | **99.5 %** | 99.1 % |

## The verdict in one line

**Physics wins the tail, the GP wins the coverage, and on the medians there is
almost nothing between them.**

Matched-mask head-to-head (signal-bin median dB / % within 3 dB), same GP
reconstruction scored on the physics arm's own mask:

| n | GP (band-A mask) | EB + GP | FEM + GP | GP (2-band mask) | EB + GP 2-band |
|---|---|---|---|---|---|
| 5 | 0.13 / 96.6 | 0.10 / 99.3 | 0.09 / 99.3 | 0.19 / 97.3 | 0.13 / 99.1 |
| 8 | 0.12 / 97.5 | 0.09 / 99.5 | 0.09 / 99.5 | 0.18 / 97.5 | 0.13 / 99.1 |
| 12 | 0.10 / 97.5 | 0.09 / 99.6 | 0.09 / 99.6 | 0.14 / 97.4 | 0.11 / 99.2 |
| 30 | 0.09 / 98.0 | 0.08 / 99.7 | 0.08 / 99.6 | 0.12 / 97.7 | 0.09 / 99.3 |

The median gap is 0.01–0.06 dB — nothing. The 3 dB fraction gap is 1.6–2.7
percentage points at every budget, which on ~140 signal bins × 93 held-out
positions is a real and consistent difference. **The physics prior does not make
the typical bin better; it removes the bad ones.** Same conclusion as the D-NS
result, arrived at independently.

## GP alone from n = 5: it plateaus early

| n | signal median (dB) | mode 1 ridge | mode 2 ridge | within 3 dB | ℓ (µm) |
|---|---:|---:|---:|---:|---:|
| 5 | 0.20 | 0.13 | 0.37 | 97.4 % | 30 |
| 6 | 0.20 | 0.15 | 0.30 | 97.6 % | 30 |
| 8 | 0.18 | 0.12 | 0.30 | 97.6 % | 30 |
| 12 | 0.14 | 0.10 | 0.20 | 97.5 % | 30 |
| 20 | 0.13 | 0.11 | 0.17 | 97.5 % | 30 |
| 30 | 0.12 | 0.10 | 0.16 | 97.8 % | 30 |

The median improves 1.7× from n = 5 to 30 and mode 2 by 2.3×, but **the 3 dB
fraction moves 0.4 points in total**. Past n ≈ 12 more positions buy the GP
essentially nothing on the tail, and it still never locates the detection node at
any budget (≈3 µm out from n = 4 to 30, see the n-sweep in
`complex-spectrum-recon-and-uncertainty.md`). PRESS picks ℓ = 30 µm at every
budget, which is stable and reassuring.

So the honest statement about the GP baseline is: **it is the only arm that
exists over the whole measurement, it is within 0.06 dB of the physics arms on
the typical bin from n = 5 onward, and its weakness is a ~2.2 % tail of bins plus
a node it never finds — neither of which more positions fix.**

## The two-band EB arm, and a correction to the published two-band result

Built `libraries/buildlib_x_cplx.py`: the solver grid of the (now archived)
`buildlib_x.py` (FLIB 185–1350 kHz, 3200 bins, 96 k₁ × 4 g, 12.8 min) with the
complex channel kept and a **graded** u grid — uniform spacing over 0.55–3.45 at
the band-A resolution would need 3300 samples and 1.5 GB, but the response only
has structure near u = 1 and u = 3.08, and everything interpolates by
`searchsorted`, so 1870 graded samples give ≤5e-4 through mode 1 and ~1.1e-3
through mode 2 for 870 MB.

Result: **k₁ = 984 ± 3 N/m at every n from 5 to 30** — a 0.6 % spread, tighter
than the published 999–1013 (1.4 %), sitting just under the mode-ratio-implied
1046 and inside the independent 1000–2154 range. Mode 2 comes in at 0.11–0.18 dB
from n = 5. Band A alone still prefers 319–394 N/m, so **Result 3 of the
two-band doc stands**: the two bands genuinely disagree about k₁ and that
disagreement is the model error.

**Correction.** `twoband.fit2g`'s default 16-point coarse grid in log k₁ is too
sparse for this cost surface. At n = 5–8 it settles at k₁ = 494–533 N/m, where a
direct scan shows k₁ = 1008 scores **0.353 against 0.607** — a strictly worse
basin the coordinate refinement cannot leave. With `coarse=(48, 15, 7)` it finds
the right basin at every n. The published two-band run's stability at n = 3–4 was
its grid happening to straddle the good basin, not robustness; anyone re-running
it on a slightly different library should expect the 500 N/m basin instead.
`rec_twoband_cplx` defaults to 48 points and says why in the docstring.

## Caveat that limits all of the above

**~85 % of the measured span is at the noise floor.** Median |Z| by region
against the floor (median |Z| above 1.5 MHz = 5.4e-4): mode 1 ridge SNR 18.5,
mode 2 ridge 6.9, everything else 0.4–1.1. All-bins dB scores therefore flatter
any smooth interpolator — two noise realisations agree to ~0.3 dB once the noise
floor is added inside the log. Every number quoted here is **signal bins only**
(median |Z| > 3× floor, ~140 of 4000 bins). The all-bins column is in the CSV
for completeness and should not be quoted.

The corollary for acquisition: there is no point scoring, or optimising a design
against, 1.5 MHz of floor. The measurement that matters is two ~40 kHz windows.

## Still open

- Nothing physical exists above ~1 MHz even with the extended library (u = 3.45
  stops at 1011 kHz at the fitted f_res). The 970.6 kHz feature and everything
  above it is GP-only territory.
- The GP's 2.2 % beyond-3 dB tail is unexplained — it is not the node alone.
- FEM still cannot join the two-band fit: it saturates at f₂/f₁ ≤ 2.94 against
  the measured 3.080, so a FEM two-band library would wedge mode 2 at the wrong
  place. `archive/fem/run_fem_etip.py` is the decider.
