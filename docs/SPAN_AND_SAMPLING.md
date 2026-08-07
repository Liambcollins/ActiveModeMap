# How much span, and how finely do you need to sample?

Two practical questions, answered from the Demo1 data and from the Euler–Bernoulli
forward model. Both were checked with `activemodemap.online.VirtualInstrument` on a
220 µm probe with 3 % noise and a 4 µm detection spot, 8 positions at rank 5, averaged
over 6 noise seeds.

## 1. Coordinates

`x = 0` is the **clamped base**; x increases toward the **free end**. The whole package
assumes this — `VirtualInstrument.measure_at` documents it, and
`lowrank._signed_zero_crossing` resolves the null to the largest x as "nearest the free
end" — so the frame must not be flipped. Where the laser is *parked* is a separate
setting, `AsylumInstrument(start_at='free_end'|'base')`.

Parking at the free end while leaving the old default (`start_at='base'`, x_start = 0)
made the loop think it was at the base and step toward the tip, which drove the spot
**off** the end of the cantilever. That is now explicit, and `x_limits_um` makes the
loop refuse any position it cannot reach.

Report the null as distance from the free end — `inst.um_from_free_end(x)`, also written
to the CSV log as `um_from_free_end`. A D-NS at x = 212.3 µm on a 220 µm probe is
7.7 µm in from the tip.

## 1b. Position is tracked in software — this is the sharp edge

There is no absolute readback in this workflow. `DoLDMove` is relative, so
`AsylumInstrument` knows where the spot is only because it performed every move
itself and added up the deltas. Two consequences:

**A new instrument object does not know where the laser is.** It takes your word
for it, from `start_at` or `x_start_um`. The active-learning loop leaves the laser
at whichever position it measured last — mid-span, not at either end. So
constructing a second instrument with `start_at='free_end'` after a run asserts
`current_x = span_um` while the spot is somewhere else entirely. Every subsequent
move inherits the difference, and the travel guard cannot help because it validates
the *believed* position. Concretely: loop ends at 138 µm on a 225 µm probe, a fresh
instrument claims 225, and a dense sweep over 225 → 107 really traverses 138 → 20,
running 87 µm past the base and onto the chip.

So: **reuse one instrument object per probe.** `close()` only withdraws and zeroes
the bias; `apply_dc_bias()` restores it, and `move_to(x)` parks the laser with one
explicit tracked, limit-checked move plus an optical image. Build a second
instrument only if you must, and then pass `x_start_um=first_inst.current_x`.

If the kernel was restarted, or the laser was moved by hand in Igor, the tracked
value is stale and nothing in software can detect it. Verify the spot optically,
then re-sync deliberately with `inst.set_position(x, confirm=True)` — which changes
bookkeeping only and refuses to run without `confirm=True`, so it cannot be
mistaken for a move.

## 2. Can you map only part of the beam?

**Yes, and for the D-NS it is better than mapping the whole thing.** The first contact
mode's displacement null sits a few micrometres in from the free end — the model puts it
at 7.3 µm from the tip for a 220 µm Multi75E-G at 1000 nN, and your own Demo1 data puts
it at 8.5 µm. So the null is inside the region you can reach, and the stiff base
contributes almost no information about it.

D-NS recovered from the free-end span only, position grid restricted to that span
(truth: x = 212.67 µm, i.e. 7.33 µm from the free end):

| span from free end | x range | recovered D-NS | error | spread over seeds |
|---|---|---|---|---|
| 20 µm | 200–220 | 212.44 | −0.22 µm | 0.08 |
| 30 µm | 190–220 | 212.54 | −0.13 µm | 0.07 |
| 40 µm | 180–220 | 213.28 | +0.61 µm | 0.33 |
| 60 µm | 160–220 | 212.73 | +0.06 µm | 0.05 |
| 80 µm | 140–220 | 212.35 | −0.32 µm | 0.08 |
| 100 µm | 120–220 | 212.65 | −0.02 µm | 0.07 |
| **120 µm** | **100–220** | **212.32** | **−0.34 µm** | **0.07** |
| 160 µm | 60–220 | 211.67 | −1.00 µm | 0.07 |
| 220 µm (full) | 0–220 | 211.52 | −1.15 µm | 0.31 |

Every span from 20 to 120 µm gives sub-micron accuracy. The *longer* spans are slightly
**worse**, because eight positions spread over the full 220 µm samples the tip region —
where the null is and where the mode shape varies fastest — more coarsely. So 120 µm of
a 220 µm probe is not a compromise for this measurement; it is the better use of the
same number of positions.

### The one thing you must not do

Set `x_grid` to the span you actually measure, not the full probe length. The low-rank
method fits a degree-(rank−1) Chebyshev polynomial over `x_grid`'s own range; extend the
grid past the data and it extrapolates. Measuring 100–220 µm but fitting on 0–220 µm
leaves the reconstructed map in the unmeasured base region wrong by a factor of ~95
(9500 % mean relative error per position), even though the D-NS itself is unaffected
because the null lies inside the measured region. The map outside the measured span is
not a prediction, it is polynomial blow-up.

If you genuinely need the mode shape over the **whole** length from a partial
measurement, that is what the physics-informed path is for: `activemodemap.loop` fits
the Euler–Bernoulli parameters and can extrapolate, at the cost of model error. The
low-rank method deliberately assumes nothing about the frequency response and therefore
cannot extrapolate in space.

## 3. Can you cover both resonances with more points instead of two runs?

**Yes — but only if you also trim the empty ends of the window.** Raising the point count
alone is not enough, because at 25 kHz – 1.774 MHz roughly half your points are spent on
spectrum that contains nothing.

Target: ≥6 points across the resonance FWHM. Your 387.5 kHz mode has Q ≈ 240, FWHM
≈ 1.6 kHz, so the step must be ≲ 270 Hz. The 1162.9 kHz mode has Q ≈ 330, FWHM ≈ 3.5 kHz,
so ≲ 580 Hz — the low mode is always the binding constraint.

| window | span | points | step | pts across 387 kHz FWHM | pts across 1163 kHz FWHM |
|---|---|---|---|---|---|
| 25 kHz – 1.774 MHz (what you ran) | 1749 kHz | 1984 | 882 Hz | **1.8** | 4.0 |
| 25 kHz – 1.774 MHz, 4× points | 1749 kHz | 3968 | 441 Hz | **3.6** | 7.9 |
| 330 kHz – 1.26 MHz | 930 kHz | 1984 | 469 Hz | **3.4** | 7.5 |
| **330 kHz – 1.26 MHz, 2× points** | 930 kHz | 3968 | **234 Hz** | **6.8** | 14.9 |
| 330 – 470 kHz (mode A alone) | 140 kHz | 992 | 141 Hz | 11.3 | — |
| 1.09 – 1.26 MHz (mode B alone) | 170 kHz | 992 | 171 Hz | — | 20.4 |
| reference dense sweep, for comparison | 195 kHz | 992 | 197 Hz | 6.0 | — |

So `330 kHz – 1.26 MHz` at ~4000 points does work for both modes at once. Note that
below 330 kHz and above 1.26 MHz your Demo1 spectra carry no features, so those points
are pure waste.

**Tune time is a separate constraint from resolution.** Points set the frequency step;
dwell per point sets whether each point has reached steady state. The mechanical response
time is `Q / (π f₀)` ≈ 0.20 ms for the 387 kHz mode and ≈ 0.09 ms for the 1163 kHz mode,
so the dwell should be comfortably longer than ~0.2 ms — otherwise the peak comes out
skewed and shifted, which biases the resonance frequency the whole D-NS extraction hangs
off. At ~1 ms per point, a 4000-point tune is ~4 s, times ~10–20 positions: not the
bottleneck compared with AutoWedge and engage.

Set `ANALYSIS_BAND_HZ` per mode even when one tune covers both. The full tune is always
saved to disk, so you can reprocess either mode afterwards without remeasuring:

```bash
python scripts/reprocess_tune_folder.py <folder> --band 330e3  470e3  --label modeA
python scripts/reprocess_tune_folder.py <folder> --band 1.09e6 1.26e6 --label modeB
```

## 4. Recommended setup for the next run

- Igor tune window `330 kHz – 1.26 MHz` at ~4000 points (234 Hz), dwell ≥ 1 ms/point,
  no auto-recenter. Or, if you prefer maximum margin, run mode A alone at
  `330 – 470 kHz` / 992 points.
- `PROBE_L_UM = 220`, `START_AT = 'free_end'`, `REACH_UM = 120` →
  `x_grid = 100 … 220 µm`.
- Run the `preview_moves()` cell and confirm every move reads "toward base" before
  engaging.
- `MIN_POSITIONS = 8`.

Expect the D-NS about 7–9 µm in from the free end.
