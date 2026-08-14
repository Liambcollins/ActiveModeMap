# FEM arm of the AL replay: physics injection pays, and Q6/Q7 answered

Setup: `Dense_Grid_A`, band A, 101 positions × 412 frequencies, calibrated
axis, ground-truth D-NS 224.09 µm. Leakage rule held — reconstructors see only
(sel_idx, A_sel, freq); the scorer alone sees withheld positions.

FEM library: AFeMulator `Multi75G` **isotropic**, 32 k₁ rungs (100–5000 N/m) ×
5 damping, free-lever gate PASS, Q 55–472 (brackets the measured 191 at every
rung), 0.00 % NaN in the aligned frame.

> **Read with:** `gp-only-baseline.md` (which halves the headline factors),
> `fitting-and-model-content.md` (the isotropic-vs-k₂=0 confound) and
> `higher-modes-and-k1.md` (which overturns the k₁ claim below).

## Positions needed to reach a target (equispaced)

| target | 3 low-rank | 1 EB | 2 EB+GP | 1b FEM | 1b FEM+GP |
|---|---|---|---|---|---|
| NRMSE ≤ 10 % | 12 | 5 | 5 | **4** | **4** |
| NRMSE ≤ 5 % | 20 | 30 | 5 | never | **5** |
| NRMSE ≤ 3 % | never | never | 6 | never | **6** |
| NRMSE ≤ 2 % | never | never | 12 | never | **8** |
| D-NS ≤ 1 µm | 20 | 10 | 8 | **4** | **4** |
| D-NS ≤ 0.5 µm | 30 | 30 | 16 | **4** | **4** |

## NRMSE (%) equispaced

| n | low-rank | EB | EB+GP | FEM | FEM+GP |
|---|---|---|---|---|---|
| 3 | 42.45 | 14.10 | 14.10 | **10.24** | 10.24 |
| 4 | 39.48 | 10.62 | 10.62 | **8.12** | 8.12 |
| 5 | 38.71 | 8.28 | 4.22 | 7.47 | **3.04** |
| 6 | 37.53 | 7.45 | 2.75 | 6.90 | **2.23** |
| 8 | 16.79 | 6.53 | 2.41 | 6.31 | **1.94** |
| 12 | 7.24 | 5.88 | 1.99 | 6.30 | **1.84** |
| 20 | 3.34 | 5.39 | 1.58 | 6.98 | **1.53** |
| 50 | 4.36 | 4.79 | 1.45 | 5.55 | **1.43** |

At n ≤ 4 the GP has < 5 revealed points and is inactive, so ±GP is identical
there by construction.

## Q6 — FEM beats EB on band A

- NRMSE n=3: 10.24 % vs 14.10 % (−27 %)
- Near-tip peak error n=3: 38.1 % vs 78.8 % (halved)
- D-NS from n=4: 0.48 µm vs 1.82 µm

FEM fits k₁ ≈ 1237 → 1154 N/m over n = 3…12 — inside the Hertz prediction
(1310 N/m) and the independent mode-A shape-fit range (1000–2154 N/m). EB fits
k₁ ≈ 387 → 351 N/m.

**Correction (see `higher-modes-and-k1.md`):** the original claim — "EB buys
its fit by distorting the contact parameter" — is wrong. k₁ is not identifiable
from band A in either model; EB reaches the measured mode-2 ratio at
k₁ ≈ 1046 N/m. Treat the k₁ discrepancy as an identifiability result. Also note
the confound: the FEM arm is isotropic (kx=ky=kz) while EB has k₂ = 0.

## Q7 — FEM still needs the GP: YES, more than EB does

Bare FEM plateaus at 5.5–7.0 % and **never reaches 5 %** even at n = 50; bare
EB reaches 4.79 % and low-rank 3.34 % at n = 20. FEM's discrepancy is larger
and more structured at high n while being much smaller at low n. FEM+GP fixes
it and is the best arm at every n ≥ 5, hitting 2 % with 8 positions vs 12 for
EB+GP.

**Recommended pairing: FEM+GP.** Bare FEM is preferable only at n ≤ 4, where
there is too little revealed data to fit a discrepancy term.

## Design sensitivity (random designs, median [IQR] NRMSE %)

| paradigm | n=5 | n=8 | n=12 | n=20 |
|---|---|---|---|---|
| low-rank | 40.8 [37.8–44.0] | 33.8 [25.1–39.9] | 8.0 [7.7–13.9] | 5.3 [4.7–8.8] |
| EB | 8.6 [8.2–9.3] | 6.7 [6.3–7.1] | 6.1 [5.5–6.6] | 5.8 [5.5–6.4] |
| EB+GP | 5.3 [5.0–6.7] | 4.0 [3.6–4.5] | 3.0 [2.6–3.4] | 2.3 [1.9–2.5] |
| FEM+GP | **5.6 [4.0–6.4]** | **3.7 [2.8–4.8]** | **2.6 [2.4–3.0]** | **2.0 [1.8–2.3]** |

At n = 8 low-rank spans a 15-point IQR; FEM+GP spans 2. Physics buys robustness
to *where* you sample, not just how much.

## Acquisition strategy is not the lever (NRMSE %, n=5)

| paradigm | equispaced | D-opt low-rank | max-variance | random |
|---|---|---|---|---|
| FEM | 7.47 | **7.31** | 8.65 | 8.57 |
| FEM+GP | **3.04** | 3.48 | 4.60 | 5.72 |
| EB+GP | **4.22** | 4.38 | 5.72 | 5.81 |

Equispaced ≈ D-optimal, both clearly beat random. Caveat — the physics arms
reuse *low-rank* designs; D-optimal under the physics posterior is untested.

## Caveat that limits the headline: bare FEM's D-NS is model-locked

Bare FEM returns D-NS = 223.606 µm for every n from 4 to 16, and across 15
random designs at n = 5 it spans only 223.60–223.64 µm. The estimate is a fixed
property of the FEM mode shape, not something the data refines; it happens to
land 0.48 µm from truth.

So "n = 4 gives sub-0.5 µm D-NS" means **"the FEM mode shape for this lever is
right to 0.48 µm"** — a bias, not convergence. It will not transfer to another
cantilever without revalidating the geometry and cannot be reduced by measuring
more. FEM+GP does move with the data (221.3–224.4 µm across designs at n = 5)
and genuinely converges to 0.04 µm by n = 30. Quote FEM+GP, not bare FEM, when
claiming D-NS accuracy.

## Cost

Median s per reconstruction: low-rank 0.002, EB 0.99, EB+GP 1.02, FEM 1.13,
FEM+GP 1.15. ~500× low-rank, still negligible against one measurement. The
surrogate library is what makes it affordable — a live FEM solve is ~11 s.

## Open

- frictionless FEM library (exports exist; resolves the lateral-spring confound)
- D-optimal / max-variance under the physics posterior
- uncertainty calibration of the physics arms
- mode B (grid must extend past 125 µm)
- two-band fitting (see `higher-modes-and-k1.md`)
