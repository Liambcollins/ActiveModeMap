"""Single source of truth for probes and datasets used in the paper.

Every notebook asks for data by key (``load("scmpitB_r2_bias")``), never by path.
Paths are relative to the data root (``config.data_root()``), which is
``D:\\ActiveModeMap`` on Liam's laptop.

``kind`` selects the loader:
  series   -- activemodemap series checkpoint: Z[condition, position, freq]
  dense    -- SCM_PIT DenseReference.npz: Z[position, freq] (+ CSV log)
  json     -- a reduced JSON summary
  tune_txt -- an Igor text-wave tune export (Frequency, Phase, Amp)
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Probe:
    key: str
    name: str
    k_N_per_m: float
    f0_free_Hz: float | None
    L_um: float | None
    clamp_offset_um: float          # stage-um minus this = distance from clamp
    bands_Hz: dict                  # name -> (lo, hi)
    qs_band_Hz: tuple               # quasi-static (off-resonance) band
    amp_divisor: float = 32.0       # AmpInvOLS = InvOLS / amp_divisor (measured, Cypher lock-in)
    notes: str = ""


PROBES = {
    "scmpitB": Probe(
        key="scmpitB", name="SCM-PIT-B (Nanosensors SCM-PIT, Pt-Ir)",
        k_N_per_m=1.697, f0_free_Hz=63.801e3, L_um=225.9, clamp_offset_um=6.1,
        # wide enough for both campaigns: R1 285/886/1706, R2 295/913/1831 kHz
        bands_Hz={"CR1": (255e3, 330e3), "CR2": (840e3, 960e3), "CR3": (1650e3, 1900e3)},
        qs_band_Hz=(15e3, 45e3),
        notes="Sep 20-21 2026, PPLN DomainsB, 500 nN. L from R1 static-shape fit; "
              "R2's own fit hits its span bound."),
    "pppcontau": Probe(
        key="pppcontau", name="PPP-CONTAu (soft Au-coated contact lever)",
        k_N_per_m=0.406, f0_free_Hz=13.649e3, L_um=450.0, clamp_offset_um=0.0,
        bands_Hz={"CR1": (50.2e3, 79.8e3), "CR2": (180e3, 220e3), "CR3": (380e3, 440e3),
                  "CR4": (650e3, 740e3), "CR5": (975e3, 1075e3)},
        qs_band_Hz=(5e3, 25e3),
        notes="Sep 17-20 2026, PPLN DomainsB, 15-250 nN. Band centres on the 1 um map (100 nN): "
              "65.0/201.8/414.1/694.5/1023.8 kHz (CR5 was mis-registered at 880-960 kHz before 2026-09-27)."),
    "scmpitA": Probe(
        key="scmpitA", name="SCM-PIT-A (SCM-PIT, Multi75G nominal geometry)",
        k_N_per_m=2.387, f0_free_Hz=None, L_um=225.0, clamp_offset_um=0.0,
        bands_Hz={"CR1": (250e3, 400e3), "CR2": (840e3, 960e3)},
        qs_band_Hz=(100.0, 5e3),
        notes="Aug 10-11 2026, PPLN, 1-1.5 uN. Draft ground truth (Dense_Grid_B). "
              "x_tip = 221.9 +/- 1.2 um on the InvOLS-corrected axis."),
    "p0": Probe(
        key="p0", name="P0 (first AL-demo probe)",
        k_N_per_m=3.63, f0_free_Hz=None, L_um=225.0, clamp_offset_um=0.0,
        bands_Hz={"CR1": (330e3, 450e3), "CR2": (1050e3, 1300e3)},
        qs_band_Hz=(100.0, 5e3),
        notes="Aug 7 2026 demos; archive only."),
}


@dataclass(frozen=True)
class Dataset:
    key: str
    probe: str
    path: str                       # relative to data root, forward slashes
    kind: str                       # series | dense | json
    description: str
    figure: str = ""                # which paper figure(s) use it
    extra: dict = field(default_factory=dict)


_D = [
    # ---------------- SCM-PIT-B, Run 2 (primary quantitative dataset) ----------------
    Dataset("scmpitB_r2_preflight", "scmpitB", "DomainsB_SCMPIT_R2/01_preflight/preflight_result.json",
            "json", "R2 pre-flight: span, CR1-3, InvOLS(x)", "6"),
    Dataset("scmpitB_r2_bias", "scmpitB", "DomainsB_SCMPIT_R2/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz",
            "series", "R2 two-domain bias survey, 8 pos x 7 V x 2 spots, 500 nN, Vac 1 V", "6, 7",
            dict(vac_V=1.0, preflight="scmpitB_r2_preflight")),
    Dataset("scmpitB_r2_wb_fast", "scmpitB", "DomainsB_SCMPIT_R2/20_wideband_fast/wideband_fast_8pos_500nN_0V_checkpoint.npz",
            "series", "R2 live wideband N=8, 0 V", "4"),
    Dataset("scmpitB_r2_wb_valid", "scmpitB", "DomainsB_SCMPIT_R2/21_wideband_validation/wideband_valid_7pos_500nN_0V_checkpoint.npz",
            "series", "R2 held-out validation, 7 never-visited midpoints", "4"),
    Dataset("scmpitB_r2_dense", "scmpitB", "DomainsB_SCMPIT_R2/75_dense_grid/dense_map_1um_500nN_0V_checkpoint.npz",
            "series", "R2 dense map, 161 pos at 1 um (192 min)", "1, 3, 5"),
    Dataset("scmpitB_r2_load_extension", "scmpitB", "DomainsB_SCMPIT_R2/40_load_ladder",
            "series", "R2 load ladder 50-500 nN x 8 pos (one checkpoint per load)", "SI"),
    Dataset("scmpitB_r2_bias_vs_load", "scmpitB", "DomainsB_SCMPIT_R2/41_bias_vs_load",
            "series", "R2 bias series at 100/300/500 nN (monotonic sweep; CR1 fit fails checks)", "SI"),
    Dataset("scmpitB_r2_ac_series", "scmpitB", "DomainsB_SCMPIT_R2/70_ac_series",
            "series", "R2 17-pt AC series 2 mV-2 V at position A, both spots", "7, SI"),
    Dataset("scmpitB_r2_images", "scmpitB", "DomainsB_SCMPIT_R2/50_quant_imaging",
            "json", "R2 quantitative CR1/QS imaging, positions A and B (QImg*.ibw in metres)", "7"),
    Dataset("scmpitB_r2_s3", "scmpitB", "DomainsB_SCMPIT_R2/analysis/s3_transfer.json",
            "json", "R2 reduced transfer function: P_qs, P_CR1, E, Q, d33 per position (reference values)", "6"),
    Dataset("scmpitB_r2_s1", "scmpitB", "DomainsB_SCMPIT_R2/analysis/s1_structure.json",
            "json", "R1 vs R2 structure: preflight, channels, dense-map nodes (reference values)", "5"),
    Dataset("scmpitB_r2_s4", "scmpitB", "DomainsB_SCMPIT_R2/analysis/s4_load.json",
            "json", "R2 load extension reduction: ladder, probe checks, bias-vs-load decomposition (reference values)", "SI"),
    Dataset("scmpitB_r2_eb_fit", "scmpitB", "DomainsB_SCMPIT_R2/analysis/r2_eb_fit.npz",
            "json", "R2 blind EB fit outputs", "5, 6"),
    Dataset("scmpitB_r2_freeair", "scmpitB", "DomainsB_SCMPIT_R2/80_closeout/Tune_SCMPIT_CLOSEOUT_FREEAIR_0900.txt",
            "tune_txt", "R2 close-out off-surface tune: tip withdrawn, electrical drive 0.05 V, laser at A", "SI",
            dict(drive_V=0.05)),
    Dataset("scmpitB_r2_closeout_walk", "scmpitB", "DomainsB_SCMPIT_R2/80_closeout/closeout_wideband_8pos_checkpoint.npz",
            "series", "R2 close-out 8-position walk, 0 V, 500 nN, minutes before the off-surface tune", "SI"),
    # ---------------- SCM-PIT-B, Run 1 ----------------
    Dataset("scmpitB_r1_preflight", "scmpitB", "DomainsB_SCMPIT/01_preflight/preflight_result.json",
            "json", "R1 pre-flight", "6"),
    Dataset("scmpitB_r1_bias", "scmpitB", "DomainsB_SCMPIT/10_bias_survey/domains_bias_checkpoint_500nN_scmpit.npz",
            "series", "R1 two-domain bias survey (same design as R2)", "6",
            dict(vac_V=1.0, preflight="scmpitB_r1_preflight")),
    Dataset("scmpitB_r1_wb_fast", "scmpitB", "DomainsB_SCMPIT/20_wideband_fast/wideband_fast_8pos_500nN_0V_checkpoint.npz",
            "series", "R1 live wideband N=8", "4"),
    Dataset("scmpitB_r1_wb_valid", "scmpitB", "DomainsB_SCMPIT/21_wideband_validation/wideband_valid_7pos_500nN_0V_checkpoint.npz",
            "series", "R1 held-out validation, 7 midpoints", "4"),
    Dataset("scmpitB_r1_freeair", "scmpitB", "DomainsB_SCMPIT/80_closeout/Tune_SCMPIT_CLOSEOUT_FREEAIR_0900.txt",
            "tune_txt", "R1 close-out off-surface tune: tip withdrawn, electrical drive 0.05 V", "SI",
            dict(drive_V=0.05)),
    Dataset("scmpitB_r1_closeout_walk", "scmpitB", "DomainsB_SCMPIT/80_closeout/closeout_wideband_8pos_checkpoint.npz",
            "series", "R1 close-out 8-position walk", "SI"),
    Dataset("scmpitB_r1_dense", "scmpitB", "DomainsB_SCMPIT/30_dense_grid/dense_map_1um_500nN_0V_checkpoint.npz",
            "series", "R1 dense map, 181 pos at 1 um (217 min)", "3, 5"),
    # ---------------- PPP-CONTAu (soft, 5 modes) ----------------
    Dataset("ppp_dense_1um", "pppcontau", "DomainsBPPPCONTAU/04_dense_map_1um_100nN/dense_map_100nN_0V_checkpoint.npz",
            "series", "Dense 346 pos at 1 um, 100 nN, 0 V (410.6 min)", "1, 3"),
    Dataset("ppp_sparse_rank6", "pppcontau", "DomainsBPPPCONTAU/05_sparse_capture_rank6/sparse_capture_100nN_0V_checkpoint.npz",
            "series", "Rank-6 AL sparse capture, 30 pos (clustered)", "3, SI"),
    Dataset("ppp_ebgp_eig", "pppcontau", "DomainsBPPPCONTAU/08_ebgp_capture_eig/ebgp_capture_100nN_0V_checkpoint.npz",
            "series", "Live EB+GP / EIG capture, 30 pos", "3, SI"),
    Dataset("ppp_coarse_ref", "pppcontau", "DomainsBPPPCONTAU/09_coarse_ref_5um/coarse_ref_5um_100nN_0V_checkpoint.npz",
            "series", "Same-session coarse reference, 70 pos at 5 um", "3, SI"),
    Dataset("ppp_wb_fast", "pppcontau", "DomainsBPPPCONTAU/10_wideband_fast_12pos/wideband_fast_12pos_100nN_0V_checkpoint.npz",
            "series", "Live wideband 12 pos in 13.8 min", "4"),
    Dataset("ppp_wb_valid", "pppcontau", "DomainsBPPPCONTAU/11_wideband_valid_15pos/wideband_valid_15pos_100nN_0V_checkpoint.npz",
            "series", "Held-out validation, 15 never-visited positions", "4"),
    Dataset("ppp_bias_15nN", "pppcontau", "DomainsBPPPCONTAU/02_domains_bias_15nN/domains_bias_checkpoint.npz",
            "series", "Two-domain bias survey, 15 nN", "7"),
    Dataset("ppp_bias_250nN", "pppcontau", "DomainsBPPPCONTAU/07_bias_survey_250nN/domains_bias_checkpoint_250nN.npz",
            "series", "Two-domain bias survey, 250 nN", "7"),
    # ---------------- SCM-PIT-A (draft ground truth) ----------------
    Dataset("scmpitA_gridB", "scmpitA", "SCM_PIT/Dense_Grid_B/DenseReference.npz",
            "dense", "Draft ground truth: 241 pos, 0.5 um commanded pitch, 1.5 uN, 4.7 h", "1-5",
            dict(log="SCM_PIT/Dense_Grid_B/DenseReference_log.csv", n_anchor=10)),
    Dataset("scmpitA_gridA", "scmpitA", "SCM_PIT/Dense_Grid_A/DenseReference.npz",
            "dense", "Earlier dense map, same lever: 101 pos at 1 um", "5, SI",
            dict(log="SCM_PIT/Dense_Grid_A/DenseReference_log.csv", n_anchor=20)),
    Dataset("scmpitA_bias", "scmpitA", "SCM_PIT/Bias Dependence/series_checkpoint.npz",
            "series", "Single-domain bias series, 7 V (-6..+6) x 8 pos, 1 uN (current Fig. 6)", "7"),
]

DATASETS = {d.key: d for d in _D}


def list_datasets(probe: str | None = None):
    """Return datasets (optionally for one probe) as a list of dicts, for display."""
    return [dict(key=d.key, probe=d.probe, kind=d.kind, figure=d.figure,
                 description=d.description, path=d.path)
            for d in _D if probe is None or d.probe == probe]
