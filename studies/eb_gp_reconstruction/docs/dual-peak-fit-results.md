<!-- 2026-08-14. Paths updated for the tidied layout; see ../MIGRATION.md. -->

# Dual-peak (two-band) fit — full results table, and one thing the earlier note left open

Re-run of the two-band fit on the complex extended library
(`results/eblib_x_cplx.npz`, built by `libraries/buildlib_x_cplx.py`), scored on
the **same setup and the same data-driven regions** as `two-band-fit-results.md`
— 265–1000 kHz, decimated ×4, 101 positions, `twoband.regions` — so every number
is directly comparable to that note. Generator: `src/run_twoband.py`. Full
table: `results/dualpeak.csv`. Figure `results/fig/d7_dualpeak.png`.

Measured: f₁ = 293.08 kHz, f₂ = 902.52 kHz, **f₂/f₁ = 3.0794**, D-NS = 223.96 µm.

## 0. The identifiability result, re-verified — EB does reach the right stiffness

Recomputed from `results/eb_f2f1.npy` (the 26-point k₁ sweep behind
`higher-modes-and-k1.md`):

| model | f₂/f₁ | at |
|---|---:|---|
| **measured** | **3.0794** | — |
| EB (2-seg, k₂ = 0) | **3.0794** | **k₁ = 1044 N/m** |
| EB, band-A fit | 2.44 | k₁ = 328 |
| EB, saturation | 3.213 | k₁ → 20000 |
| FEM isotropic, best over the whole ladder | 2.942 | k₁ = 5000 |
| FEM frictionless | 2.66 | saturates |

**EB reaches the measured ratio at k₁ = 1044 N/m** — essentially Hertz (1310)
and inside the independent range (1000–2154). The FEM number is new here:
computed directly from `eigen/frequencies_hz` filtered to `mode_types == 1`
across the isotropic ladder, it tops out at 2.942 and cannot reach 3.079 at any
k₁ (cached as `results/fem_f2f1.npy`).

Mechanism, from the earlier note: over k₁ = 328 → 1291 N/m, f₁ moves 3.8 % while
the ratio moves 28 % — **the ratio is 7.3× the more sensitive channel**. Band A
carries the mode shape, band B carries the stiffness.

**This is now on the deck.** Slide 2 previously showed EB's band-A k₁ = 374 N/m
beside FEM's 1254 with no caveat, which let a reader re-derive the discredited
"EB is 3.4× too soft" claim; slide 3 cited "contact stiffnesses 3× apart" as
evidence the two models are independent, borrowing force from a gap that is
mostly a band-A identifiability artefact. Both are corrected, and the status
slide lists it as a correction.

## 1. Fitted contact stiffness k₁ (N/m)

| n | A only | A+B, 1 gain | **A+B, per-band** |
|---|---:|---:|---:|
| 3 | 391.9 | 1065.3 | **974.3** |
| 4 | 391.6 | 759.9 | **1003.3** |
| 5 | 391.9 | 1014.5 | **972.4** |
| 6 | 389.8 | 1011.2 | **971.8** |
| 8 | 386.0 | 1532.2 | **968.5** |
| 10 | 371.6 | 1557.6 | **997.9** |
| 12 | 350.1 | 1556.8 | **1000.3** |
| 16 | 369.6 | 1579.6 | **1000.2** |
| 20 | 339.2 | 1571.1 | **1000.2** |
| 30 | 318.4 | 1578.6 | **1000.6** |

**989 ± 15 N/m over n = 3–30, 3.5 % spread.** Reproduces the published
999–1013 (1.4 %) — same basin, slightly wider scatter; the wide-axis variant in
`wideband-arm-comparison.md` gave 984 ± 3 over n = 5–30. All three agree on
~1000 N/m, just under the ratio-implied 1044.

Band A alone sits at 318–392 N/m and **drifts downward with n** — more data makes
the band-A-only estimate worse, the signature of fitting a biased model harder.
One global gain lands near the right value but swings 760–1580 (published
755–1571 ✓).

## 2. Where each arm puts the mode-2 peak (error in kHz)

| n | A only | A+B, 1 gain | A+B, per-band |
|---|---:|---:|---:|
| 3 | −144.8 | +0.4 | **−1.3** |
| 4 | −144.8 | −24.1 | **0.0** |
| 8 | −155.8 | +20.6 | **−0.9** |
| 12 | −178.1 | +20.6 | **0.0** |
| 30 | −189.4 | +21.0 | **0.0** |

The per-band arm is **within one 437 Hz bin at every budget from 3 to 30**
(published: "within one frequency bin at n = 4" ✓ — 757.7 kHz / −144.8 kHz for
the A-only arm at n = 4 matches exactly). The A-only error *grows* with n, from
−145 to −189 kHz, for the same reason its k₁ drifts.

Model f₂/f₁ at the fitted k₁ = **3.0715** against the measured 3.0794 — 0.26 %
low. The ratio is essentially satisfied, which is what pins k₁.

## 3. NEW — the per-band gain measures the drive asymmetry

The published note could only say the per-band gain "absorbs" a relative
amplitude error. With the gains written out:

> **gain ratio = −9.51 ± 1.24 dB → mode 2 couples 2.99× weaker than mode 1.**

`antiresonance-drive-and-observable.md` independently derived **3.11×** from the
`Domains1` domain-flip data. Two unrelated routes — a fitted nuisance parameter
here, a physical experiment there — agreeing to 4 %. That promotes the
distributed-electrostatic-drive explanation from plausible to corroborated, and
means the per-band gain is measuring something, not hiding something.

## 4. Held-out median |dB| by region

| n | arm | mode-1 ridge | mode-2 ridge | antiresonance |
|---|---|---:|---:|---:|
| 8 | A only | **0.63** | 18.98 | 22.40 |
| 8 | A+B, 1 gain | 2.43 | 4.82 | 26.38 |
| 8 | A+B, per-band | 3.46 | 1.37 | 24.35 |
| 8 | **A+B, per-band + GP** | **0.20** | **0.26** | **10.68** |
| 20 | A only | **0.25** | 20.66 | 22.25 |
| 20 | A+B, per-band | 2.01 | 0.86 | 24.94 |
| 20 | **A+B, per-band + GP** | **0.13** | **0.15** | **5.66** |

Physics-only, the published trade-off is confirmed: including mode 2 buys ~18 dB
on the mode-2 ridge and costs 1.4–2.8 dB on the mode-1 ridge, at every n.

## 5. The trade-off is not real once the GP is there

The earlier note tested physics-only arms and left the antiresonance open.
Adding the discrepancy GP to the per-band arm:

| n | mode-1 ridge | mode-2 ridge | antiresonance | whole band | ℓ (µm) |
|---|---:|---:|---:|---:|---:|
| 5 | 0.53 | 0.52 | 15.53 | 1.10 | 15 |
| 8 | 0.20 | 0.26 | 10.68 | 0.50 | 8 |
| 12 | 0.19 | 0.19 | 6.86 | 0.40 | 8 |
| 20 | 0.13 | 0.15 | 5.66 | 0.31 | 8 |
| 30 | 0.13 | 0.13 | 4.78 | 0.29 | 8 |

**Both ridges at once, to 0.13–0.53 dB, from n = 5.** "Mode 2 costs you mode 1"
is a statement about the bare forward model only; the GP absorbs the inter-band
inconsistency and you keep both. The deployable arm has no trade-off.

**And the antiresonance is not unfixable.** The published note recorded 19–25 dB
"in every arm" and left it open — every one of those arms was physics-only. With
the GP it falls 15.5 → 10.7 → 6.9 → 4.8 dB from n = 5 to 30: slowly, still the
worst region by an order of magnitude, but data-limited rather than
model-limited. Restated: **the forward model cannot place the antiresonance and
no amount of two-band information changes that; what fixes it is positions,
~1 dB per doubling of n.**

## 6. What stands from the earlier note

- **Result 1** (per-band gain stabilises k₁): confirmed, 989 ± 15 N/m.
- **Result 2** (~18 dB on mode 2 for 1.5–3.8 dB on mode 1): confirmed
  physics-only; dissolved once the GP is added.
- **Result 3** (the two bands genuinely disagree about k₁ — 390 vs 1000 — so the
  residual is a *frequency* disagreement, not amplitude bookkeeping): **stands,
  and is the main open physics question.** Front-runner is still contact
  location: a contact at ~224 µm rather than the assumed point reconciles the
  band-A and mode-ratio estimates at ~1000. Checkable in the optical frames.
- **Isotropic library destabilises the two-band fit** (667–2486 N/m): not
  re-tested; `eblib_ic_x` was not rebuilt. Worth doing — the variant benchmark
  found EB k₂ = k₁ + a 100 N/m compliant cone is the one variant that satisfies
  the mode ratio *and* Hertz simultaneously, at k₁ ≈ 1310.
  (`archive/eb_variants/buildlib_ic.py`.)

## 7. Caveat on the optimiser — read before re-running

`twoband.fit2g`'s default `coarse=(16, 11, 5)` is too sparse in log k₁. At
n = 5–8 it settles at k₁ = 494–533 N/m where a direct scan gives cost 0.607
against **0.353 at k₁ = 1008** — a strictly worse basin the coordinate
refinement cannot leave. Everything above uses `coarse=(48, 15, 7)`. The
published run's stability at n = 3–4 with 16 points was its grid straddling the
good basin; on a slightly different library the same code lands in the 500 N/m
basin instead. Anyone comparing k₁ across library versions should check the
coarse grid first.
