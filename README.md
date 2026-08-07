# ActiveModeMap

**Rapid mode-shape mapping in dynamic AFM by active learning.**

The spatial shape of a vibrating AFM cantilever — its mode shape as a function of
detection position *x* and drive frequency *f* — determines what a dynamic-AFM
measurement reports: the contact stiffness in CR-AFM, and the null spot / blind
spot operating points in interferometric PFM. Measuring the full position–frequency
map by stepping the detection laser and sweeping frequency at every position is
slow and must be redone per probe, per load. ActiveModeMap recovers the full map
from a handful of laser positions chosen by active learning, because the map is
low-dimensional.

It provides **two complementary reconstructions of the same low-dimensional map**:

- **Physics-informed** — fits the few parameters of a fast Euler–Bernoulli
  contact-resonance model. Fewest positions (3–4 in simulation), returns physical
  parameters (contact stiffness, drive ratio), can extrapolate; but only as good as
  the model. A Gaussian-process discrepancy term keeps it accurate under realistic
  model error.
- **Model-light (low-rank)** — assumes only that the mode shape is smooth in space,
  fitting the map as a low-rank spatial object with **no** assumption about the
  frequency response. Sharp resonance/antiresonance features are preserved and model
  error is impossible; needs a few more positions (~6). **This is the method used on
  the instrument**, where beam theory is not exact.

See `paper/` for the full write-up and the trade-off between the two.

## Install

```bash
pip install -r requirements.txt          # core: numpy scipy scikit-learn matplotlib
# on the instrument PC also:  pywin32  igor2  pandas  pillow
pip install -e .                          # optional, to import activemodemap anywhere
```

Python 3.9+. The core (simulation, reconstruction, offline analysis) is
cross-platform; only `activemodemap.asylum` needs Windows + Igor Pro.

## Layout

```
activemodemap/            core library
  forward_model.py        two-pathway Euler–Bernoulli contact-resonance model
  virtual_afm.py          simulated instrument with noise + laser-spot blur
  inference.py            physics-informed Bayesian parameter posterior
  acquisition.py          EIG / variance / spot-targeted position selection
  baselines.py            random / equispaced / physics-free GP
  hybrid.py               GP discrepancy + model-free spot extraction (model error)
  loop.py                 physics-informed active-learning driver
  lowrank.py              model-light low-rank reconstruction (+ online manager)
  online.py               Instrument abstraction, VirtualInstrument, live plotting
  asylum.py               Asylum/Igor control + AsylumInstrument (Windows only)
notebooks/
  00_reference_dense_sweep_asylum.ipynb   original dense-sweep automation (reference)
  01_active_loop_dryrun.ipynb             the loop on a VirtualInstrument (no hardware)
  02_reconstruct_dense_data.ipynb         low-rank reconstruction on bundled real data
  03_run_on_instrument.ipynb              >>> run the full loop on the AFM <<<
scripts/                  benchmark + figure-generation scripts
data/example_1000nN/      bundled 30-position dense sweep (Multi75E-G / PPLN, 1000 nN)
paper/                    manuscript draft
figures/                  generated figures
```

## Quick start

**Off the microscope** — understand and validate the loop:

```bash
jupyter notebook notebooks/01_active_loop_dryrun.ipynb     # simulated instrument
jupyter notebook notebooks/02_reconstruct_dense_data.ipynb # real bundled data
```

**On the microscope** — `notebooks/03_run_on_instrument.ipynb`. Before running,
in Igor: approach and find the contact resonance; set a **fixed, wide tune window**
covering the resonance *and* the antiresonance across your scan (no auto-recenter);
park the laser at the start of the span. The notebook then drives the loop —
pick position → move laser → optical image → AutoWedge + InvOLS → engage → tune →
read the complex spectrum → update reconstruction → pick the next position — and
stops when the D-NS confidence interval is small enough, saving the reconstruction,
the raw spectra, and a per-position log.

The online loop is hardware-agnostic: it only needs an object with
`measure_at(x_um) -> (freq_Hz, Z_complex, meta)`. `VirtualInstrument` and
`AsylumInstrument` both implement it, so the identical loop code runs in simulation
and on the instrument.

```python
from activemodemap import LowRankModeMap
mm = LowRankModeMap(x_grid_um, rank=5, dns_ci_tol_um=1.0)
for _ in range(max_positions):
    x = mm.next_position()
    freq, Z, meta = instrument.measure_at(x)
    mm.add_measurement(x, freq, Z)
    if mm.n >= mm.min_positions:
        rec = mm.reconstruct()
        if mm.converged():
            break
```

## Notes

- The bundled example maps the **displacement null spot (D-NS)** from a single-domain
  tune series. For the **electrostatic blind spot (D-ESBS)**, measure both PPLN domain
  orientations per position and separate the piezoresponse and electrostatic channels
  (difference and sum); the same low-rank reconstruction applies to each.
- `notebooks/00_...` is the original dense-sweep automation, kept as the reference and
  as the source of the shared Asylum control code in `activemodemap/asylum.py`.

Center for Nanophase Materials Sciences, Oak Ridge National Laboratory.
