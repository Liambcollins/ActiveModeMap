# paper_analysis — shared analysis code for the FMM / quantitative CR-PFM paper

One package (`fmmpaper`) and one notebook per paper figure. Every figure and every
number in the paper should be reproducible from here, from the raw checkpoints on disk.

```
paper_analysis/
  fmmpaper/            the shared layer (import it from any notebook)
    config.py          paths: data root, activemodemap repo, EB repo, FMM code zip
    registry.py        probes + datasets by key  <- the ONLY place file paths live
    io.py              loaders -> Series(Z[condition, position, freq]), one phase convention
    axis.py            InvOLS position ruler (Grid B), pre-flight InvOLS(x), static-shape fit
    spectra.py         peaks, Q, mode profiles, nodes, detection null (D-NS)
    domains.py         two-domain bias decomposition, V_cpd, E(x), d33
    recon.py           GP / low-rank reconstruction, selectors, held-out + cross-capture scores
    published.py       reads the draft's archived benchmark tables (EB+GP, oracle)
    plotting.py        one figure style (fonts, palette, column widths), save PNG + PDF
    results.py         save numbers behind each figure to results/*.json + *.csv
  notebooks/
    00_data_overview                         working
    01_fig1_three_probes_and_drift           planned (keys wired)
    02_fig2-3_sparse_reconstruction_gridB    working
    03_fig3_how_many_and_where               planned
    04_fig4_live_fmm                         working
    05_fig5_physics_from_sparse              planned
    06_fig6_transfer_function_d33            working
    07_fig7_bias_maps_and_imaging            planned
    SI_calibration_and_reproducibility       planned
  tools/               notebook sources (nb_*.py) + build_notebooks.py
  tests/               regression tests against the archived numbers
  figures/  results/   outputs (regenerated; safe to delete)
```

## Setup (Windows, once)

```bat
cd D:\ActiveModeMap\paper_analysis
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt
pip install -e C:\Users\lz1\Documents\Github\ActiveModeMap_repo\ActiveModeMap
jupyter lab notebooks
```

No path configuration is needed on Liam's laptop: the data root defaults to the parent
of this folder (`D:\ActiveModeMap`), and the repo paths default to `Documents\Github`.
Elsewhere, copy `config.local.example.json` to `config.local.json` and edit, or set
`FMM_DATA_ROOT`, `FMM_AMM_REPO`, `FMM_EB_REPO`, `FMM_CODE_ZIP`.
Check with `python -c "import fmmpaper; print(fmmpaper.config.describe())"`.

## Using it

```python
import fmmpaper as F
from fmmpaper import axis, domains, recon, plotting as fp
F.available()                                   # what's registered and on disk
s = F.load("scmpitB_r2_bias")                   # Series: Z[14 conditions, 8 positions, 58816 f]
s.where(bias_V=0, spot=1); s.band("CR1")        # condition indices, frequency mask
t = domains.transfer_table(s, axis.invols_interpolator(F.load("scmpitB_r2_preflight")), vac_V=1.0)
```

Add a dataset by adding one `Dataset(...)` line to `registry.py`; nothing else changes.

## Conventions (enforced on load)

| quantity | convention |
|---|---|
| `Z` in checkpoints | complex, raw lock-in **volts**; metres = V × InvOLS / 32 (÷32 measured for the Cypher lock-in) |
| `QImg*.ibw` images | `AmplitudeRetrace` already in **metres** — never apply InvOLS/32 again |
| phase | model convention (falls through resonance). Pre-2026-09-17 files are Igor-raw and are conjugated on load (`meta['phase_converted']`) |
| position | stage µm as logged; distance from clamp = `x_um − probe.clamp_offset_um`; Grid B also has `meta['x_true_um']` (InvOLS ruler) |
| frequency bands | per probe in `registry.PROBES[...].bands_Hz` (CR1…CRn) and `qs_band_Hz` |

## Verified against the archive

| check | result |
|---|---|
| Grid B InvOLS ruler | x_true = 0.843 x + 37.48 µm, 0.52 µm rms (draft S1) |
| Grid B D-NS (dense) | 224.11 µm (draft) |
| GP arm, mode-A NRMSE, N = 3…20 | identical to `regen.pkl` (max diff 0.000) |
| R2 E(x), d33_QS, V_cpd per position | identical to `s3_transfer.json` (machine precision) |
| R2 d33 medians (interior) | QS 8.81, CR1÷EB 8.31, CR1÷Q 11.49 pm/V (spread 6.6×) |
| Live PPP-CONTAu, 12 → 15 pos | 0.169 NRMSE, 97.8 % within 3 dB at rank 8 (report: 0.169, 98 %) |
| Live SCM-PIT-B R2, 8 → 7 pos, package code | 0.191 NRMSE, 98.9 % within 3 dB at rank 5 (report re-implementation: 0.193, 93 %) |

Run `pytest -q tests` after any change to `fmmpaper`.

## Known gaps

- **EB+GP is not re-run here.** The draft's physics arm needs the two-segment EB library
  binding (EB-Solver-CResonance, several GB). Notebook 02 overlays the archived EB+GP values.
- **The laptop copy of `activemodemap` is GitHub `580c491`.** The 2026-09-17 instrument-side
  changes (phase fix in `asylum`, `ebgp_capture`, fitted ζ/setback in `PhysicsPosterior`) are
  not in it. This package handles phase itself, so loaders are unaffected; the EB physics arm
  on September data needs those changes committed first.
- The blind EB fit (`fit_eb_r2.py`) is read from `r2_eb_fit.npz`, not refitted.
- Q here uses parabolic peak interpolation (R2 mean 238 ± 5); the manuscript quotes 233 ± 5.
- Draft's mode-A oracle values are not in the archived `oracle.pkl` (that has full-band only).
