"""Planned notebooks: data keys wired, analysis steps listed, cells ready to fill."""
from nbbuild import SETUP

STUBS = {
"01_fig1_three_probes_and_drift": ("Why the transfer function must be measured, and fast (Fig. 1)", """
**Panels:** concept/workflow schematic; dense Z(x, f) for three probes (2, 5 and 3 modes);
drift over 9.5 h; measurement-time bar.

**Data keys:** `scmpitA_gridB`, `ppp_dense_1um`, `scmpitB_r2_dense`, `ppp_coarse_ref`,
`ppp_wb_fast`, `ppp_wb_valid` (drift = same positions re-measured hours apart).

**Steps**
1. Load the three dense maps; plot with `fp.map_db` on a common dB scale, CR labels.
2. Drift floor: NRMSE between `ppp_coarse_ref` (00:00) and the 12/15-position captures (10:09, 10:37)
   at matching positions -> expect 0.293–0.295; show it is a frequency shift, not a gain.
3. Time bar: dense 241 pos / 4.7 h, 346 pos / 410.6 min, 161 pos / 192 min vs FMM 7–18 min.
""", ["scmpitA_gridB", "ppp_dense_1um", "scmpitB_r2_dense", "ppp_coarse_ref"]),
"03_fig3_how_many_and_where": ("How many positions, and where (Fig. 3)", """
**Panels:** error vs N with oracle and random IQR (Grid B, from notebook 02); per-mode
crossover vs node count on the PPP-CONTAu 1 µm grid; EIG vs equispaced wideband divergence;
D-NS error vs N.

**Data keys:** `ppp_dense_1um`, `ppp_sparse_rank6`, `ppp_ebgp_eig`, `ppp_coarse_ref`,
`scmpitB_r1_dense`, `scmpitB_r2_dense`.

**Steps (Phase 2a, 2c)**
1. Budget sweep per CR band on `ppp_dense_1um` with `recon.budget_sweep` (GP, low-rank).
2. EB+GP per band: needs the physics arm (PhysicsPosterior with the 2026-09-17 phase fix,
   which is not in the laptop copy of the package; see README).
3. Low-rank from the EIG design (`ppp_ebgp_eig` positions) vs equispaced -> divergence.
4. Repeat the Grid B benchmark on the SCM-PIT-B dense maps (third ground truth).
""", ["ppp_dense_1um", "ppp_ebgp_eig", "scmpitB_r2_dense"]),
"05_fig5_physics_from_sparse": ("Physics recovered from sparse spectra (Fig. 5)", """
**Panels:** k1 and f0 stability vs N (published, Grid B); blind EB fit inputs/outputs (R2);
node shifts R1 -> R2 as a contact-state readout; SVD of the beam-model residual.

**Data keys:** `scmpitB_r1_dense`, `scmpitB_r2_dense`, `scmpitB_r2_s1`, `scmpitB_r2_eb_fit`,
published tables (`fmmpaper.published.load('regen')`).

**Steps (Phase 2d)**
1. Node positions from both dense maps with `spectra.nodes_from_profile` per CR band;
   compare with `scmpitB_r2_s1` (CR2 132.9 -> 146.9 um, CR3 96.9/167.9 -> 110.9/178.9).
2. Port `fit_eb_r2.py` (R2 analysis snapshot) into `fmmpaper.eb` and refit both contact states.
3. SVD residual: requires the EB library binding (EB-Solver-CResonance).
""", ["scmpitB_r2_s1", "scmpitB_r2_eb_fit"]),
"07_fig7_bias_maps_and_imaging": ("Bias-resolved mode maps and quantitative imaging at resonance (Fig. 7)", """
**Panels:** P, b, V_cpd along the lever per mode (stiff vs soft); |b|/|P| vs mode number;
V_cpd from spectroscopy vs images; CR1 d33 maps at positions A and B, uncompensated vs
V_cpd-compensated; SNR at 1 V vs 30 mV.

**Data keys:** `scmpitB_r2_bias`, `ppp_bias_15nN`, `ppp_bias_250nN`, `scmpitA_bias`,
`scmpitB_r2_images` (QImg*.ibw, calibrated metres; read with `igor2`), `scmpitB_r2_ac_series`.

**Steps**
1. `domains.decompose_series` per probe and mode band; |b|/|P| per mode (PPP-CONTAu CR1–CR5).
2. Image loader for `QImg*.ibw` (AmplitudeRetrace already in metres -- do NOT apply InvOLS/32),
   domain mask from phase, equalisation ratio at V_cpd (expect 1.011 / 1.012).
3. AC series: fit |Z| = sqrt((k V_ac)^2 + n0^2) per domain (expect k 56/83 mV/V, n0 570 uV in R2).
4. Single-domain draft Fig. 6 (`scmpitA_bias`): V0 vs V_cpd discussion, now with two-domain data.
""", ["scmpitB_r2_bias", "scmpitA_bias"]),
"SI_calibration_and_reproducibility": ("SI: calibration, units, load ladder, reproducibility", """
**Sections:** InvOLS ruler (S1), tip position (S2), units convention (raw V vs metres, ÷32,
QPDI detection), InvOLS ageing (~10 % systematic), load ladder (CR1 saturates, E ×2.45),
AC linearity floor, R1 vs R2 (including the two overturned conclusions).

**Data keys:** `scmpitA_gridB` (log), `scmpitB_r1_preflight`, `scmpitB_r2_preflight`,
`scmpitB_r2_load_extension`, `scmpitB_r2_bias_vs_load`, `scmpitB_r2_ac_series`.
""", ["scmpitB_r1_preflight", "scmpitB_r2_preflight"]),
}


def cells_for(title, plan, keys):
    return [
        ("md", f"# {title}\n\n*Status: planned — data keys wired, analysis to fill in.*\n{plan}"),
        ("code", SETUP),
        ("code", "F.available().query(\"key in " + repr(keys) + "\")"),
        ("code", "# data = {k: F.load(k) for k in " + repr(keys) + "}\n# TODO: analysis steps above"),
    ]
