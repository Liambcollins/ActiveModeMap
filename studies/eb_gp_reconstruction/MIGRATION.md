# Migration map: `studies/physics_informed_al` → `studies/eb_gp_reconstruction`

Nothing was deleted. The folder was renamed, demoted arms moved under
`archive/`, duplicated material moved to `../../_to_delete/`, and every
container-absolute path replaced with a `config.py` lookup.

## Why the rename

The old name led with the part of the study that did not work: **acquisition
chose nothing**. Every number in the deck and in `results/` uses equispaced
positions, and every adaptive strategy tried was worse (GP-only max-variance
12.94 % vs 3.41 % equispaced at n = 8). What the study actually established is a
reconstruction result — two-segment EB, EB + discrepancy GP, and GP alone — so
the folder is named for that.

To revert: `git mv studies/eb_gp_reconstruction studies/physics_informed_al`.
Nothing outside the folder referenced the old name (checked with grep across the
repo); the only internal reference was one line in `docs/fem-runs-wanted.md`.

## Moved to `archive/`

| old path | new path | why |
|---|---|---|
| `src/run_fem.py` | `archive/fem/run_fem.py` | FEM arm demoted |
| `src/femcmp.py` | `archive/fem/femcmp.py` | merges the FEM tables |
| `libraries/build_fem_library.py` | `archive/fem/build_fem_library.py` | FEM library |
| `libraries/build_femlib_cplx.py` | `archive/fem/build_femlib_cplx.py` | FEM library, complex |
| `libraries/run_fem_ladder.py` | `archive/fem/run_fem_ladder.py` | drives AFeMulator |
| `libraries/run_fem_etip.py` | `archive/fem/run_fem_etip.py` | tip-compliance diagnostic |
| `figures/d_al.py` | `archive/fem/d_al.py` | FEM vs EB vs low-rank |
| `figures/d_models.py` | `archive/fem/d_models.py` | renders the FEM geometry |
| `libraries/buildlib_ic.py` | `archive/eb_variants/buildlib_ic.py` | EB k₂ = k₁ + compliant cone |
| `libraries/buildlib_x.py` | `archive/eb_variants/buildlib_x.py` | superseded by `buildlib_x_cplx.py` |
| `figures/d_esdrive.py` | `archive/eb_variants/d_esdrive.py` | electrostatic-drive investigation |
| `figures/d_twoband.py` | `archive/superseded/d_twoband.py` | log-amp only; replaced |
| `figures/d_wideband.py` | `archive/superseded/d_wideband.py` | band-A only; replaced |
| `figures/d_figs.py` | `archive/superseded/d_figs.py` | grab-bag of early deck figures |

The low-rank arm has **no file to move**: it lives in `physrec.rec_lowrank` and
is still produced by `src/run_phys.py` for the baseline table. It is demoted in
emphasis only.

## Moved to `../../_to_delete/` (review, then delete)

| what | size | why |
|---|---|---|
| `_source/` | 3.1 MB | a copy of the whole study, stale by 5 files |
| `_source.zip` | 2.5 MB | the same copy, zipped |
| `../../commit_activemodemap_fixes.ps1` / `.sh` | — | one-off commit scripts left at the repo root |
| `../../commit_tune_wait_fix.ps1` | — | same |

These were **moved, not deleted** — this session cannot delete files on the
device. Check `_to_delete/` holds nothing you want, then remove the folder.

## New in this pass

| path | what |
|---|---|
| `src/cplxrec.py` | complex-field arms + predictive σ (updated: `fit_band`, two-band entry point) |
| `src/twoband.py` | joint two-band fit, per-band gain |
| `src/run_twoband.py` | dual-peak results, 4 arms × 10 budgets |
| `src/run_wideband.py` | full measured band, 25 kHz – 1.775 MHz |
| `libraries/buildlib_x_cplx.py` | extended complex EB library, graded u grid, 870 MB |
| `figures/d_twoband.py` | new: mode-ratio identifiability, k₁ stability, the trade-off |
| `figures/d_wideband.py` | new: coverage, per-frequency error, the tail metric |
| `figures/d_tf_shapes.py` | new: transfer functions and mode shapes vs raw data |
| `figures/ornl_logo.png` | needed by `reporting/build_deck_cplx.py` |
| `results/dualpeak.csv`, `results/wideband.csv`, `results/fem_f2f1.npy` | this pass's numbers |
| `results/fig/d6_wideband.png` … `d9_shapes.png` | this pass's figures |
| `compare_to_main.sh` | diffs this tree against `origin/main` in your own clone |

## Path portability

Eight files carried absolute paths from the container they were written in
(`/home/claude/amm/...`, `/mnt/user-data/uploads/...`). All are now resolved
through `config.py`:

```
/home/claude/amm/out/<f>                  ->  {config.OUT}/<f>
/home/claude/amm/out/fig                  ->  config.FIG
.../SCM_PIT/Dense_Grid_A                  ->  config.DENSE_GRID_A
.../SCM_PIT/Bias Dependence/...           ->  config.BIAS_SERIES   (new)
.../EB-Solver-CResonance/...              ->  config.EB_REPO, config.EB_GEOMETRY
.../fem_eb_comparison/data_ladder         ->  config.FEM_LADDER
```

Two `config.py` fixes worth knowing about:

1. `add_eb_to_path()` now adds **both** `EB_REPO/src` and
   `EB_REPO/fem_eb_comparison/py`. It previously added only `src/`, so
   `libraries/buildlib.py` — which imports `geometry` and `eb_models` — worked
   only when the working directory was `fem_eb_comparison/py`.
2. `BIAS_SERIES` and `LIBRARIES` were added, so no script names a library file
   by literal path.

Every script now locates `config.py` by walking up from its own file until it
finds it, instead of assuming a fixed depth. That is what lets the archived
scripts keep working one directory deeper.

## `src/physrec.py` change

The module-level `np.load(OUT/'eblib.npz')` is now guarded by an
`os.path.exists` check, and `_peak()` was hoisted above the guard with an
optional `_FW` argument. Effect: `import physrec` succeeds on a clone where no
library has been built yet, or where only the complex library exists. The
alignment code inside the guard is unchanged, so every published number is
reproduced bit-for-bit.
