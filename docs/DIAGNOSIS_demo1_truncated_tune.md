# ActiveModeMap — Demo1 run: why it picked the wrong resonance

Diagnosis of the 2026-08-07 `03_run_on_instrument` run in
`D:\User Data\Liam\ActiveModeMap\Demo1`.

## The one-line answer

`activemodemap/asylum.py`, `get_tune_data()`:

```python
n = fw.GetDimensions()[0]      # returns 2, not 1984
```

Element 0 of Igor's COM `GetDimensions()` is the *dimension count*; element 1 is the
row count. `get_gmv()` a few lines above has always correctly used `[1]`. Because the
tune waves are one-dimensional, `[0]` is 2, so every "spectrum" the loop handed to the
reconstruction was **the first two frequency bins of the tune** — 24999.998 Hz and
25882.055 Hz — and nothing else.

That is the narrow frequency axis. `LowRankModeMap` is written to adopt the
instrument's own frequency grid from the first measurement (`freq_grid=None`), so the
2-point axis became the common grid for the whole run.

## How the evidence lines up

Every number in the run is explained by that single line.

The `ActiveModeMap_result.npz` frequency axis is exactly two values, and they are
bit-for-bit the first two bins of the saved tune file: 24999.998046875 and
25882.0546875 (the float32 values Igor stores; the text file rounds them to
24999.998 and 25882.055). The `resonance_freq_Hz` column in the log only ever takes
those same two values, because `argmax` over a 2-point amplitude array can only return
bin 0 or bin 1. The logged `q_factor` is 28.343 or 29.343 — differing by exactly 1.0,
because `Q = f_res / bandwidth` and the only bandwidth a 2-point array can produce is
one frequency step, 882.06 Hz: 24999.998/882.06 = 28.343 and 25882.055/882.06 = 29.343.
The measured amplitudes stored in the npz are all ~9 × 10⁻⁴ V, which is the noise floor
at 25 kHz, and finally D-NS came out as 120.00 µm with a CI of exactly 0.00 µm — the
last grid point, with zero uncertainty, which tripped the `dns_ci < 1.0 µm`
convergence test immediately at the 7th position.

The bug is inherited from `afm_laser_sweep_automation_v4` and is present in
`notebooks/00_reference_dense_sweep_asylum.ipynb` too. It was harmless there: it only
corrupted a log column. The bundled `data/example_1000nN` log records
`resonance_freq_Hz = 254319.95` and `254123.13` — again exactly bins 1 and 0 of that
tune window — when the real resonance in those files is at **357.1 kHz**. The dense-sweep
analysis (notebook 02) reads the saved `.txt` files rather than the COM arrays, so
nobody noticed. Notebook 03 is the first place the COM arrays became the actual data.

## Your data is fine — the run is fully recoverable

`save_tune()` uses Igor's own `Save/T` command, so the `Tune_*.txt` files on disk are
complete: 1984 points from 25.0 kHz to 1774.1 kHz, and both resonances you were after
are there, with the amplitude at each varying strongly with laser position exactly as a
mode shape should.

Reprocessing the seven saved tune files:

| band | resonance | result |
|---|---|---|
| 330–470 kHz (159 pts) | 387.53 kHz | **D-NS = 111.5 µm**, single clean branch crossing |
| 1090–1260 kHz (193 pts) | 1162.85 kHz | two nulls, at **12.7 µm** and **117.5 µm** |

Mode B legitimately has two nulls in the span, so a single D-NS number is ambiguous for
it — use `dns_branch()` (added below) to see all crossings rather than trusting the one
number `dns_from_map` picks.

The quoted bootstrap CIs (0.1 µm and 0.2 µm) are far too optimistic and should not be
reported. The bootstrap draws its noise scale from the fit residual, and with 7
positions at rank 5 there are only 2 residual degrees of freedom, so the residual
underestimates the noise badly. This is also why `MIN_POSITIONS = 6` (= rank + 1) will
always "converge" almost immediately regardless of data quality — worth raising to 8+.

## Two more things to fix before the next run

**Frequency resolution.** A 25 kHz – 1.774 MHz window at 1984 points is an 882 Hz step.
The 387.5 kHz contact resonance has a FWHM of ~1.6 kHz (Q ≈ 240), so it is spanned by
about **two** points. Your reference dense sweep used 254–449 kHz at 992 points = 197 Hz
step and had 6 points across the FWHM. Two points is not enough to locate the
resonance/antiresonance line shape that the D-NS depends on. Tune one mode at a time:
330–470 kHz for the 387 kHz mode is ~150 Hz per point at 992 points. Run the loop
twice rather than trying to catch both modes in one window.

**Analysis band.** `dns_from_map` searches for the antiresonance notch within
`window_frac × (freq.max() − freq.min())` of the resonance. With `window_frac = 0.35`
that is ±68 kHz for your reference window but ±612 kHz for a 1.75 MHz window, which
makes the search meaningless. Even with the point-count bug fixed, a multi-mode window
needs to be cropped before it reaches the fit. `AsylumInstrument` now takes
`analysis_band_Hz=(lo, hi)` for this; the full tune is still saved to disk.

## What changed in the code

`activemodemap/asylum.py`

- `get_tune_data()` no longer trusts `GetDimensions()[0]`; a new `_wave_npnts()` helper
  takes the largest reported dimension (documented, with the reasoning) and raises if
  it is implausibly small.
- New `read_tune_txt()` parses the file Igor saves. `tune_eigenmode()` now **saves
  first and reads the file**, falling back to COM only if that fails. The file is
  guaranteed complete, and it avoids ~6000 COM round-trips per position, so each
  position is also faster. `read_tune_txt` is pure Python, so it imports off the
  instrument PC.
- `extract_resonance_from_tune()` takes an optional band, interpolates the peak
  parabolically so `f_res` is not quantised to the frequency step, and warns when the
  peak sits on a window edge or the FWHM spans fewer than 5 points.
- `AsylumInstrument(analysis_band_Hz=..., min_points=32)`: crops what the fit sees,
  and **raises** rather than proceeding on a short spectrum. The log now records
  `n_tune_points`, `tune_f_lo_Hz`, `tune_f_hi_Hz` and `n_fit_points` so a truncated
  read is visible in the CSV.

`activemodemap/lowrank.py`

- `LowRankModeMap.add_measurement()` rejects spectra shorter than 32 points, so this
  class of failure cannot silently reach a reconstruction again.
- `_dns_branch_crossing()` now discards positions where the amplitude minimum lands on
  the *edge* of the search window — there is no notch there, and pinning `fa` to the
  boundary manufactured fake crossings. This is what produced a spurious mode-B null at
  66.5 µm, in the middle of the beam where the map shows nothing.
- New `dns_branch()` returns *all* branch crossings plus `fa(x)`, for inspecting a map
  before trusting a single number.
- `d_optimal_order()`: while fewer than `rank` positions are selected, `Bᵀ B` is
  rank-deficient and its determinant is zero for every candidate, so the greedy choice
  was decided by floating-point noise. That is why your run measured both 20 and 21 µm
  and both 99 and 100 µm — two of seven positions spent on nearly redundant
  measurements. It now scores each candidate at the highest basis order the current set
  can support, with a small ridge.

`notebooks/03_run_on_instrument.ipynb`

- `ANALYSIS_BAND_HZ` parameter, passed to `AsylumInstrument`.
- The loop prints the tune window it actually received and asserts on the first
  position that the spectrum is long enough and the peak is not on a band edge.
- Setup notes now cover the resolution requirement and one-mode-per-run.
- Prints all branch crossings alongside the D-NS estimate.

`scripts/reprocess_tune_folder.py` (new) — rebuild any run from its saved tune files,
with `--band` to select the mode. Reports the resolution across the FWHM and all branch
crossings.

## Suggested next run

1. In Igor set the tune window to 330–470 kHz at ~1000 points, no auto-recenter.
2. `ANALYSIS_BAND_HZ = (330e3, 470e3)`, `MIN_POSITIONS = 8`.
3. Watch the first position's printout: it should say ~159–1000 points and a strongest
   response at ~387.5 kHz. If it says 2 points, something is still wrong upstream.
4. Repeat with the Igor window at 1090–1260 kHz and `ANALYSIS_BAND_HZ` to match for the
   second mode.

Expect the D-NS for the 387 kHz mode to land near 111 µm — that is what your existing
data says once it is read correctly.
