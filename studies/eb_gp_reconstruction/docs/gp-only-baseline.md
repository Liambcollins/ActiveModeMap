# GP alone vs physics-informed — and why this qualifies the headline

`physrec.rec_gp_only` is `rec_eb_gp` with the physics mean removed and nothing
else changed: same `T(A) = log(A + F)` transform, same noise floor estimated
from the revealed block only, same isotropic squared-exponential over position
shared across all frequencies, same PRESS length-scale selection, same noise
rule. The mean function is a per-frequency constant fitted to the revealed
positions. So any gap between the arms is attributable to the mean function.

Its length-scale grid is **wider** than the discrepancy GP's — (1.5, 2.5, 4, 8,
15, 30, 60, 120) µm vs (4 … 120) — the baseline is given more freedom, not less.

A second variant, `gp_oracle`, picks the length scale by minimising the
**held-out** error. That is an **oracle diagnostic, not a method** — it exists
only to separate "the GP cannot represent this field" from "PRESS cannot pick a
length scale from 3 points".

## Held-out complex-map NRMSE (%), equispaced

| n | 3 low-rank | GP only | GP only (oracle ℓ) | 1 EB | 1b FEM | 2 EB+GP | 1b FEM+GP |
|---|---|---|---|---|---|---|---|
| 3 | 42.45 | 63.60 | 22.49 | 14.10 | **10.24** | 14.10 | 10.24 |
| 4 | 39.48 | 36.10 | 9.83 | 10.62 | **8.12** | 10.62 | 8.12 |
| 5 | 38.71 | 6.82 | 6.82 | 8.28 | 7.47 | 4.22 | **3.04** |
| 8 | 16.79 | 3.41 | 3.41 | 6.53 | 6.31 | 2.41 | **1.94** |
| 12 | 7.24 | 2.46 | 2.35 | 5.88 | 6.30 | 1.99 | **1.84** |
| 20 | 3.34 | 1.82 | 1.82 | 5.39 | 6.98 | 1.58 | **1.53** |
| 50 | 4.36 | 1.64 | 1.64 | 4.79 | 5.55 | 1.45 | **1.43** |

## Positions needed

| target | 3 low-rank | GP only | 1 EB | 1b FEM | 2 EB+GP | 1b FEM+GP |
|---|---|---|---|---|---|---|
| NRMSE ≤ 10 % | 12 | 5 | 5 | **4** | 5 | **4** |
| NRMSE ≤ 5 % | 20 | 7 | 30 | never | **5** | **5** |
| NRMSE ≤ 3 % | never | 10 | never | never | **6** | **6** |
| NRMSE ≤ 2 % | never | 20 | never | never | 12 | **8** |
| D-NS ≤ 1 µm | 20 | 10 | 10 | **4** | 8 | **4** |
| D-NS ≤ 0.5 µm | 30 | 12 | 30 | **4** | 16 | **4** |

## Four findings

**1. The low-rank baseline was too weak, and the headline was inflated by it.**
"3× fewer positions for a usable map" was 4 vs 12 against low-rank; against a
plain GP it is **4 vs 5**. Sub-micron D-NS holds up better — 4 vs 10, not
4 vs 20. Any reviewer would reach for a GP first.

**2. A plain GP comprehensively beats BARE physics from n = 5 onward.** GP-only
needs 5/7/10/20 positions for the four NRMSE targets; bare EB manages
5/30/never/never and bare FEM 4/never/never/never. The discrepancy GP is doing
most of the work at moderate n. Physics premium (GP-only ÷ FEM+GP): 2.25× at
n = 5, 1.76× at 8, 1.34× at 12, 1.19× at 20, 1.14× at 50.

**3. Where physics genuinely wins.**
- **n ≤ 4:** GP-only 63.6 / 36.1 % vs FEM 10.2 / 8.1 %. Even oracle-ℓ manages
  only 22.5 / 9.8 % — at three points there is no substitute for a model.
- **D-NS:** GP-only cannot produce a node estimate at n = 3 and is 28.9 µm out
  at n = 4; FEM gives 0.48 µm at n = 4.
- **Design robustness:** at n = 5 random designs, GP-only 21.6 [11.1–40.7] vs
  FEM+GP 5.6 [4.0–6.4] %.
- **Interpretable parameters:** a stiffness vs a length scale.

**4. GP-only's small-n collapse is partly a selection failure.** PRESS picks
ℓ = 1.5 µm at n = 3 where ℓ = 30 µm is best (63.6 → 22.5 %); at n = 4 it picks
8 µm where 15 µm is best (36.1 → 9.8 %). From n ≥ 5 PRESS and oracle coincide.

Cost: 0.0011 s per reconstruction vs 1.15 s for FEM+GP — ~1000× cheaper, and no
FEM ladder to build.

## The defensible claim

> Against a plain GP baseline, physics-informed reconstruction is worth about
> 1.15–2.25× lower error at n ≥ 5, and the advantage grows sharply below n = 5,
> where a GP has too few points either to fit or to tune. For locating the
> detection node to sub-micron the physics arms need 4 positions against the
> GP's 10–12, and they remove most of the design sensitivity.

## Reproduce

`python src/run_gp.py 24` → `results/gp_bench.csv`; figure `figures/d_gp.py`.
