# scripts/eb_gp_dense — 2026-09-17 offline tests on the PPP-CONTAu dense sweep

`prep.py`, `eb_gp_test.py`, `make_fig.py`, `eb_zeta.py`, `eb_phase.py`, `run_conj.py`,
`final_fig.py` are the scaffolding that found the three issues below. They subclass
the package to test each change in isolation and are SUPERSEDED by package options:

| scaffolding here                      | now in the package                                   |
|---------------------------------------|------------------------------------------------------|
| `SingleDomainPosterior` / `SingleDomainHybrid` | `PhysicsPosterior` and `HybridSurrogate` accept data without a "minus" key |
| `EBZetaModel` / `ZetaPosterior`       | `EBForwardModel` 6-entry theta; `PhysicsPosterior(fit_zeta=True)` |
| `EBPhaseModel` / `PhasePosterior`     | `PhysicsPosterior(analytic_gain=True)` (complex gain, solved per evaluation) |
| `np.conj(Zg)` in `run_conj.py`        | `asylum.tune_to_complex` default `PHASE_SIGN = -1`, `check_convention=True` |
| assumed `tip_setback_um`              | `PhysicsPosterior(fit_geometry="setback")` |

Regression + feature tests for the package changes: `python tests/test_physics_refactor.py`.
Findings and numbers: project docs `eb-gp-on-pppcontau-dense-2026-09-17` and
`eb-phase-convention-and-zeta-2026-09-17`.
