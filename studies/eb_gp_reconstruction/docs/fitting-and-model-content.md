# What is actually fitted, and what the two models actually contain

Verified against source: `libraries/buildlib.py`, `eb_models.py`,
`geometry.py`, `src/physrec.py`, `run_fem_sweep.py`, and the `input/`
attributes of a ladder export.

## The fit: four numbers per reveal step

`physrec.fit_eb` — identical machinery for EB and FEM, since both libraries are
reduced to the same resonance-aligned surrogate.

| parameter | how | bounds |
|---|---|---|
| `log k1` contact stiffness | searched | library range: EB 150–20 000 N/m, FEM 100–5000 N/m |
| `f_res` contact resonance | searched | ±2 % of the peak in the **revealed** spectra |
| `log g` damping | searched | EB air damping 5000–20 000 s⁻¹ (Q 113–283); FEM fluid damping 10^4.6–10^5.3 N s/m³ (Q 55–472) |
| `gain` | **closed form** | one scalar: `mean(log A_obs − M)` over bins with A > 3F |

**Objective:** MSE in `T(A) = log(A + F)` over revealed positions × bins.
`F` = median of per-position minima, revealed block only (the band contains
exact zeros; raw log|Z| runs to −690 and a handful of bins dominated the
objective).

**Window:** bins where the revealed peak profile clears 10 % of its maximum —
32 of 412 bins, 285.8–299.3 kHz. Both models sit ~8 dB above the measured
off-resonance floor (a mode-2 placement problem, see
`higher-modes-and-k1.md`), so fitting the full band drives k₁ to the library
edge.

**Search:** coarse grid 16 × 11 × 5 = 880 evaluations, then 3 rounds of
coordinate-wise bounded Brent. Refitted from scratch at every reveal step.

**No per-position or per-frequency freedom.** Four numbers reconstruct
101 × 412 = 41 612 complex values — why the physics arms work at n = 3, and
why a GP needs more data. The ±GP arms then add one length scale (LOO-PRESS)
and one noise level on the residual, shared across all frequencies.

## What the EB model actually contains

`EBConfig(probe=P, model='2seg', phi=deg2rad(11), k1=k1, k2=0., g=g)` with
`ContactParams(k1, k2, c1=0, c2=0, psi=0, k_cone_lat=inf, k_cone_ax=inf)`.

**HAS:** 11° tilt; a genuine 15.76 µm free overhang (contact at the interior
point as moment/shear jump conditions, z″(L)=z‴(L)=0 at the free end); tip
height H = 13.53 µm as a moment arm; trapezoidal cross-section
(w_top 35.1 µm, w_bot 19.7 µm), E = 170 GPa, ρ = 2330 kg/m³.

**LACKS:** lateral contact spring (k₂ = 0); contact dashpots (c₁ = c₂ = 0 —
damping is distributed air damping only); tip-cone compliance (rigid); axial
taper (prismatic section held constant); lumped tip mass; anything but
flexural modes.

Note: `eb_models.py` calls `g` "presentational only" — stale; the library
genuinely varies with g (Q 283 → 113 across the grid).

## The Q6 confound

`run_fem_sweep.LATERAL`: frictionless → kx=ky=0 (EB k₂=0); isotropic →
kx=ky=kz (EB k₂=k₁). The benchmark FEM arm is **isotropic** while EB has
**k₂ = 0**, so "FEM beats EB" conflates geometry fidelity with the lateral
contact spring. All 160 frictionless exports exist; build with

```
python build_fem_library.py --data <ladder> --name Multi75G \
    --lateral frictionless --out femlib_frictionless.npz
```

and re-run `src/run_fem.py` against it. The complementary fix on the EB side is
`libraries/buildlib_ic.py` (k₂ = k₁ + compliant cone — see
`model-variants-mode-ratio.md` for why the cone is required, not optional).

## One more shared approximation

Both models are compared to the data on **vertical displacement**; the optical
lever senses **beam slope**. The FEM exports carry `xz_slope` as a separate
channel — for mode 1 the normalised profiles are close (0.63 vs 0.58 of channel
max), so the substitution is defensible, but `xz_slope` is the more faithful
observable.
