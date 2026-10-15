"""Regression tests for activemodemap.smm against the archived SCM-PIT R2 survey.

Set SMM_R2_DIR to the DomainsB_SCMPIT_R2 folder (default: D:\\ActiveModeMap\\DomainsB_SCMPIT_R2).
Archived reference numbers (paper_analysis, fixed-bin convention): delta = 0.635 V,
corrected QS d_eff 7.58 pm/V (sd 0.8 %), E_meas/E_LOO 0.94-1.05 at interior stops.
"""
import json
import os

import numpy as np
import pytest

R2 = os.environ.get("SMM_R2_DIR", r"D:\ActiveModeMap\DomainsB_SCMPIT_R2")
CKPT = os.path.join(R2, "10_bias_survey", "domains_bias_checkpoint_500nN_scmpit.npz")
pytestmark = pytest.mark.skipif(not os.path.exists(CKPT), reason="R2 survey not on disk")


@pytest.fixture(scope="module")
def r2():
    from activemodemap.series import load_checkpoint
    s3 = json.load(open(os.path.join(R2, "analysis", "s3_transfer.json")))
    pf = json.load(open(os.path.join(R2, "01_preflight", "preflight_result.json")))
    return dict(series=load_checkpoint(CKPT), invols={r["x_um"]: r["invols"] for r in s3["rows"]}, pf=pf)


def test_map_route_reproduces_archive(r2):
    from activemodemap.smm import SMMConfig, analyse_survey
    res = analyse_survey(r2["series"], SMMConfig(), f_cr1_Hz=r2["pf"]["cr1_Hz"], invols_by_x=r2["invols"], verbose=False)
    assert abs(res["delta"]["delta_V"] - 0.635) < 0.08                      # peak-tracked vs fixed-bin differ slightly
    d = res["deff_qs_pm_per_V"]
    assert abs(np.nanmedian(d) - 7.58) < 0.15
    assert np.nanstd(d) / np.nanmedian(d) < 0.03
    ratios = res["emap"]["loo_ratio"][1:-1]                                  # interior stops
    assert np.all((ratios > 0.85) & (ratios < 1.15))
    assert res["recommend"]["recommended"] is not None
    assert abs(res["esbs"]["x_esbs_um"] - 232.0) < 1 and res["esbs"]["raw_error_pct"] < 4


def test_be_conversion_synthetic(tmp_path):
    from activemodemap.smm.imaging import _sho, be_deff
    rng = np.random.default_rng(0)
    f = np.linspace(260e3, 340e3, 93); f0, Q, P, c = 300e3, 250.0, 0.02, 0.004 * np.exp(0.3j)
    Z = np.zeros((16, 16, f.size), complex); dom = np.zeros((16, 16), bool); dom[:, :8] = True
    for i in range(16):
        for j in range(16):
            Z[i, j] = ((P if dom[i, j] else -P) + c) * _sho(f, f0, Q) + 0.002 * (rng.standard_normal(f.size) + 1j * rng.standard_normal(f.size))
    p = tmp_path / "be.npz"; np.savez(p, freq_Hz=f, Z=Z)
    r = be_deff(str(p), E=250.0, invols_m_per_V=1e-6, drive_V=1.0)
    assert abs(r["deff_pm_per_V"] - P * 1e-6 / 32 * 1e12 / 250) < 0.05 and abs(r["Q_domain1"] - Q) / Q < 0.05


def test_replay_workflow(tmp_path, r2):
    import matplotlib
    matplotlib.use("Agg")
    from activemodemap.smm import SMMConfig, stages
    cfg = SMMConfig()
    cfg.campaign.data_dir = str(tmp_path)
    cfg.campaign.virtual = True
    cfg.campaign.replay_checkpoint = CKPT
    cfg.campaign.replay_invols_json = os.path.join(R2, "analysis", "s3_transfer.json")
    cfg.capture.n_positions = 8
    inst = stages.connect(cfg)
    pf = stages.preflight(inst, cfg)
    assert 60 <= pf["x_lo_um"] <= 80 and abs(pf["cr1_Hz"] - 295.3e3) < 2e3
    series = stages.capture(inst, cfg, live=False)
    assert series["x_um"].size == 8
    res = stages.analyse(cfg, series, show=False)
    assert abs(np.nanmedian(res["deff_qs_pm_per_V"]) - 7.58) < 0.3
    plan = stages.plan_imaging(cfg, be_positions_um=[150.0])
    kinds = [s["kind"] for s in plan["stops"]]
    assert kinds == ["qs", "be"] and abs(plan["stops"][0]["stage_um"] - 232.0) < 1 and plan["stops"][0]["raw_error_pct"] < 5
    assert 250 < plan["stops"][1]["E_map"] < 350 and plan["stops"][1]["loo_validated"]
    log = stages.run_imaging(inst, cfg)
    assert log["frames"] == [] and len(log["stops"]) == 2
    m = stages.closeout(inst, cfg)
    assert m["plan"]["stops"][1]["E_map"] > 0


def test_coarse_to_fine_order():
    from activemodemap.smm.stages import coarse_to_fine
    xs = coarse_to_fine(np.arange(8, dtype=float)[::-1])
    assert sorted(xs) == list(range(8)) and set(xs[:3]) == {7.0, 0.0, 4.0}


def test_config_roundtrip(tmp_path):
    from activemodemap.smm import SMMConfig, load_config
    cfg = SMMConfig(); cfg.capture.bias_V = [-6, -3, 0, 3, 6]; cfg.probe.qs_band_Hz = (10e3, 40e3)
    p = cfg.save(str(tmp_path / "c.json"))
    c2 = load_config(p)
    assert c2.capture.bias_V == [-6, -3, 0, 3, 6] and c2.probe.qs_band_Hz == (10e3, 40e3)
    cfg.capture.bias_V = [0, 3]
    with pytest.raises(ValueError):
        cfg.check()
