# Which model variants can reach the measured mode ratio?

Diagnostic run 2026-08-12, deciding which variants earn a full benchmark arm.
Measured f2/f1 = 3.080 (flexural modes 292.97 and 902.41 kHz). Reference
limits: clamped-pinned 3.240, clamped-clamped 2.757.

| variant | saturation / value | reaches 3.080? | at k1 |
|---|---|---|---|
| EB k2=0, rigid tip | -> 3.213 | yes | ~1046 N/m |
| EB k2=k1, rigid tip | saturates ~2.78 | **no** | — |
| EB k2=k1, cone 1000 N/m | -> ~2.95 | no | — |
| EB k2=k1, cone 300 N/m | -> ~3.09 | marginal | ~4000 N/m |
| **EB k2=k1, cone 100 N/m** | -> ~3.16 | **yes** | **~1310 N/m = Hertz** |
| FEM frictionless | saturates 2.66 | no | — |
| FEM isotropic | 2.94 @ k1=5000, still rising | not in range | — |

Three conclusions.

1. **Adding the lateral spring alone makes the mode ratio WORSE.** EB k2=k1
   with a rigid tip saturates at 2.78 — the rigid lateral constraint
   over-stiffens the rotation at the contact and caps the ratio below the
   measurement. This mirrors FEM: frictionless 2.66, isotropic 2.81-2.94.
2. **The lateral spring must act through a compliant cone.** EB with k2=k1 and
   k_cone_lat ~ 100 N/m hits the measured ratio at k1 = 1310 N/m — the Hertz
   value — simultaneously satisfying the two independent constraints. This is
   the physically consistent EB variant, and `libraries/buildlib_ic.py` builds
   its library (`eblib_ic.npz`).
3. **FEM's saturation needs a diagnosis, not a guess.** Its elastic tip column
   is the suspect (same mechanism as the EB cone, but its effective stiffness
   is fixed by e_tip = 130 GPa and the STL cone geometry, not fittable).
   `libraries/run_fem_etip.py` is the 12-export test: if raising e_tip lifts
   f2/f1 toward 3.1, the tip column explains it.

Caveat: k_cone = 100 N/m was SELECTED to satisfy (ratio, Hertz) jointly — it is
calibrated on band-B information, not independently measured. The honest test
is whether the eblib_ic arm improves *held-out band-A* reconstruction without
touching that number, which is what the benchmark does.

## Benchmark outcome (ebic_bench.csv, 2026-08-12)

Held-out NRMSE %, equispaced — the new arm slots between EB and FEM:

| n | EB k2=0 | EB iso-cone | FEM (isotropic) | EB+GP | EB-ic+GP | FEM+GP |
|---|---|---|---|---|---|---|
| 3 | 14.10 | 12.53 | **10.24** | 14.10 | 12.53 | 10.24 |
| 5 | 8.28 | 7.43 | 7.47 | 4.22 | 3.30 | **3.04** |
| 8 | 6.53 | 5.88 | 6.31 | 2.41 | 2.25 | **1.94** |
| 20 | 5.39 | 5.05 | 6.98 | 1.58 | 1.56 | **1.53** |
| 50 | 4.79 | 4.20 | 5.55 | 1.45 | 1.44 | **1.43** |

Three take-aways:

1. **The lateral spring + cone helps EB everywhere** (14.1 -> 12.5 at n=3;
   4.8 -> 4.2 at n=50) and its GP arm nearly matches FEM+GP from n=8 on.
2. **FEM keeps a real low-n edge over the now-matched EB variant** (10.2 vs
   12.5 at n=3) — with the lateral condition roughly equalised, that residual
   is a cleaner geometry statement than the original Q6 (still isotropic-FEM
   vs cone-calibrated EB, so not perfectly clean).
3. **Band-A k1 identifiability is unchanged:** the iso-cone fit returns
   k1 ~ 470 N/m where its own mode ratio implies ~1310. No variant recovers a
   physical k1 from band A alone — the two-band fit remains the fix.
