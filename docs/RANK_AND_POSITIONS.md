# How few positions can you get away with?

Short answer: **6**, at rank 4. Not fewer, and the way to get there is by lowering
`RANK`, not by lowering `MIN_POSITIONS` on its own.

## Why `MIN_POSITIONS` cannot simply be reduced

`reconstruct_map` fits a Chebyshev basis of dimension `rank` along position, so:

- **npos < rank** — underdetermined. `reconstruct_map` silently clamps with
  `rank = min(rank, len(sel_idx))`, so you get a lower-rank fit than you asked for
  without being told. `LowRankModeMap` now raises instead.
- **npos == rank** — exactly determined. The residual is *identically zero*, so
  `sig = 0`, `std = 0`, and the bootstrap CI is `0`. `converged()` then fires on a
  fit that has never been tested against anything. This is what produced the
  original `D-NS = 120.00 ± 0.00 µm`.
- **npos == rank + 1** — 1 residual degree of freedom. Technically a CI, but from a
  single residual, so still not trustworthy.
- **npos >= rank + 2** — 2+ dof. The smallest honest configuration, and the new
  default (`min_positions = rank + 2`).

So the floor is set by `rank`. Lower the rank and the floor drops with it.

## But rank is not free — and more is not better

D-NS error against the Euler–Bernoulli forward model, 225 µm probe, 118 µm of reach
from the free end, 3 % noise, 4 µm spot, `npos = rank + 2`, mean over 6–8 noise
seeds. True D-NS 7.42 µm from the free end.

| rank | npos | D-NS error | spread over seeds |
|---|---|---|---|
| 2 | 4 | +0.84 µm | 0.09 |
| **3** | **5** | **+3.16 µm** | 0.05 |
| **4** | **6** | **+0.04 µm** | 0.07 |
| 5 | 7 | −0.79 µm | 0.07 |
| 6 | 8 | −0.88 µm | 0.22 |

Rank 3 is the trap: it looks stable — the seed-to-seed spread is the *smallest* in
the table — while being 3 µm wrong. Precision without accuracy. Rank 6 starts to
overfit noise into the null region and the spread triples.

Holding across conditions (mean |error|, `npos = rank+2`):

| condition | rank 3 | rank 4 | rank 5 | rank 6 |
|---|---|---|---|---|
| baseline (225 µm probe, 118 µm reach) | 3.15 | **0.04** | 0.79 | 0.91 |
| noisier (8 %) | 3.20 | **0.13** | 0.37 | 1.01 |
| short reach (60 µm) | 0.78 | **0.20** | 0.34 | 0.55 |
| bigger spot (8 µm FWHM) | 3.90 | 0.89 | **0.05** | 0.20 |
| longer probe (450 µm) | 0.82 | 0.52 | **0.23** | 1.26 |

Rank 4 wins three of five, rank 5 the other two, and the two are within about
1 µm everywhere. **Use rank 4 or 5. Never rank 3.**

On the real Demo1 data (7 positions, mode A) the same pattern holds — ranks 3, 4
and 5 cluster at 7.6–8.5 µm from the free end, while ranks 2 and 6 jump to 1.5 and
2.4 µm, which are not credible for a first contact mode.

## The bootstrap CI is not your uncertainty

The CI propagates measurement noise through a fit whose rank is *assumed correct*.
It is therefore blind to the largest error term: the choice of rank. On Demo1 the
CI at rank 4 is 0.43 µm, while re-fitting the same data at ranks 3–5 spreads the
answer over 0.93 µm — about twice as much.

`mm.rank_sensitivity()` does this for free, with no extra measurements:

```
  D-NS vs rank (7 positions):
    rank 3:   112.33 um
    rank 4:   112.40 um
    rank 5:   111.47 um
  median 112.33 um, spread across neighbouring ranks 0.93 um
```

Quote the spread, not the CI. For Demo1 mode A that makes the result
**7.7 ± 0.5 µm from the free end**.

It defaults to `rank-1, rank, rank+1`. Scanning wider is not more conservative, it
is just wrong: including rank 2 and rank ≥ n−1 mixes in configurations you would
never use and inflated the Demo1 spread from 0.9 µm to 7.9 µm. Pass `ranks=` once
per probe to confirm rank 4–5 really is the plateau, then leave it at the default.

## Recommended settings

```python
RANK          = 4    # or 5; NOT 3
MIN_POSITIONS = 6    # = RANK + 2
DNS_CI_TOL_UM = 1.0
```

That is 6 positions instead of the previous 8, and — at least in simulation — more
accurate. Convergence also now requires the estimate to have stopped moving across
three successive reconstructions (`stable_over=3`), not merely to have a small CI,
so a run that stops at 6 has actually earned it.
