# Archive: demoted arms, still runnable

Nothing here is broken and nothing here is wrong. These arms were moved out of
the study's main line because the headline result is now carried by three arms —
two-segment EB, EB + discrepancy GP, and GP alone — and keeping five arms in the
foreground made the comparison harder to read, not easier.

Each script still runs. Every one locates `config.py` by walking up from its own
file, so being one directory deeper costs nothing:

```bash
python archive/fem/run_fem.py            # from the study root, or anywhere
python archive/eb_variants/buildlib_ic.py
```

`src/` and `figures/` are on `sys.path` automatically, so `import physrec`,
`from run_phys import prep` and `import ornl` all resolve as before.

---

## `fem/` — the geometry-faithful FEM arm

AFeMulator beam-network model of the actual Multi75G geometry, driven over a
32-rung k₁ ladder and stitched into a surrogate library with the same
`(k₁, damping, u, position)` axes as the EB library, so it drops into the same
reveal loop.

**Why demoted, given it is the most accurate arm** (1.94 % NRMSE at n = 8 vs
2.41 % for EB+GP):

- It cannot be rebuilt without a running GUI application answering HTTP on
  `127.0.0.1:8050`. The EB library is one `python` command. For a result other
  people are meant to reproduce, that difference decides it.
- Its advantage over EB+GP is 0.5 percentage points on amplitude, and **zero**
  on the two things that matter most: both land 0.15 µm from the detection node,
  and both fail phase identically (41.4° vs 42.4°).
- On the mode ratio it is actively worse. FEM-isotropic saturates at
  f₂/f₁ = 2.942 over the whole ladder — **no k₁ reaches the measured 3.0794** —
  while EB reaches it at k₁ ≈ 1044 N/m. The mode ratio is what identifies the
  contact stiffness, so the arm that cannot span it cannot do the measurement.
  (Suspect: the elastic tip column. `run_fem_etip.py` is the 12-export test.)
- Its lateral condition is isotropic (kx = ky = kz) while EB is built with
  k₂ = 0, so the published FEM-vs-EB comparison conflates geometry with the
  lateral contact spring (`docs/fitting-and-model-content.md`).

| file | what |
|---|---|
| `run_fem_ladder.py` | drives AFeMulator: 32 k₁ × 5 damping × 2 lateral |
| `run_fem_etip.py` | 12-export diagnostic: does tip compliance cap f₂/f₁? |
| `build_fem_library.py` | ladder exports → resonance-aligned log-amp library |
| `build_femlib_cplx.py` | the same, keeping Re/Im (26 s; 32/32 rungs, f_res 312.4 → 375.9 kHz, Q 472 → 55) |
| `run_fem.py` | the FEM / FEM+GP benchmark arm |
| `femcmp.py` | merged comparison tables, crossover points |
| `d_al.py` | FEM vs EB vs low-rank, four panels |
| `d_models.py` | the two forward models side by side, FEM drawn from the real STL |

Needs `AMM_FEM_LADDER` (defaults to `EB_REPO/fem_eb_comparison/data_ladder`).

## `eb_variants/` — EB models that are not the headline model

| file | what | why demoted |
|---|---|---|
| `buildlib_ic.py` | EB with k₂ = k₁ and a 100 N/m compliant tip cone | built to test whether the lateral spring explains the EB/FEM gap; results in `results/ebic_bench.csv` and `docs/model-variants-mode-ratio.md`. Never rebuilt with the complex channel — doing so and re-running the two-band fit against it is a live open item |
| `buildlib_x.py` | extended EB library, log-amplitude only | superseded by `libraries/buildlib_x_cplx.py`, which adds Re/Im on a graded u grid (870 MB instead of 1.5 GB) |
| `d_esdrive.py` | contact drive alone vs contact + distributed electrostatic load | the PFM drive-model investigation. Relevant to the unexplained 42° phase error, but it is a separate question from the reconstruction result |

## `superseded/` — figures replaced by newer versions

| file | replaced by | difference |
|---|---|---|
| `d_twoband.py` | `figures/d_twoband.py` | the old one is log-amplitude and predates the `coarse=(48,15,7)` fix, so its small-n k₁ values sit in the bad basin near 500 N/m |
| `d_wideband.py` | `figures/d_wideband.py` | the old one covers band A only; the new one covers the full 25 kHz – 1.775 MHz measured span and shows each arm's library coverage |
| `d_figs.py` | `figures/d_cplx_spectra.py` and others | the early grab-bag: `d_truth2d`, `d_2drecon`, `d_caveat`, `d_q6`, `d_q7`. Their PNGs are still in `results/fig/` and the published deck still uses them |

## The low-rank baseline has no file here

It lives in `physrec.rec_lowrank` and `src/run_phys.py` still produces its
column, because it is the model-light reference the physics is measured against.
It is demoted in emphasis only: it needs 12 positions for 10 % NRMSE and 20–30
to locate the node, against 8 and 4 for the physics arms, and it flattered the
physics badly enough that the GP-only arm had to be built as a fair baseline
(`docs/gp-only-baseline.md`).

`strat_maxvar` also still selects on the low-rank reconstructor's predictive
variance regardless of which arm is being scored — which is one reason every
adaptive acquisition strategy underperformed equispaced sampling.
