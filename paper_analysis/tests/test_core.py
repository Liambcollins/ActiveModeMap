"""Regression tests: the package must reproduce the archived analysis numbers.

    pytest -q tests            # skips any test whose data is not on disk
"""
import numpy as np
import pytest

import fmmpaper as F
from fmmpaper import axis, domains, recon, spectra


def _need(*keys):
    av = F.available().set_index("key").on_disk
    missing = [k for k in keys if not av.get(k, False)]
    if missing:
        pytest.skip(f"data not on disk: {missing}")


def test_domain_decomposition_synthetic():
    rng = np.random.default_rng(1)
    V = np.array([-9, -6, -3, 0, 3, 6, 9.0])
    P, b, vcpd = 2.0 + 0.5j, 0.4 - 0.1j, 1.1
    z1 = P + b * (V - vcpd) + 1e-6 * rng.standard_normal(7)
    z2 = -P + b * (V - vcpd)
    d = domains.decompose(V, z1, V, z2)
    assert np.isclose(d["P"], P) and np.isclose(d["b"], b)
    assert np.isclose(d["v_cpd"], vcpd)
    # flip is defined on the 0 V intercepts (archived convention), so it is exactly
    # 180 deg only when V_cpd = 0
    d0 = domains.decompose(V, P + b * V, V, -P + b * V)
    assert abs(d0["flip_deg"] - 180) < 1e-6


def test_gridB_axis_and_dns():
    _need("scmpitA_gridB")
    g = F.load("scmpitA_gridB")
    slope, off = g.meta["axis_fit"]
    assert abs(slope - 0.843) < 0.002 and abs(off - 37.5) < 0.3      # draft Sec. S1
    f0, v = spectra.on_resonance_profile(g.freq_Hz, g.Z[0], g.probe.bands_Hz["CR1"])
    assert abs(spectra.dns_signed(g.meta["x_true_um"], v)["dns"] - 224.11) < 0.05


def test_gridB_gp_matches_draft():
    _need("scmpitA_gridB")
    g = F.load("scmpitA_gridB"); x = g.meta["x_true_um"]; Z = g.Z[0]
    sel = recon.select_equispaced(x, 6); h = recon.held_out(len(x), sel)
    r = recon.rec_gp(x, sel, Z[sel])
    assert abs(recon.nrmse(r["Zrec"][h], Z[h], g.band("CR1")) - 6.611) < 0.01


def test_r2_transfer_matches_archive():
    _need("scmpitB_r2_bias", "scmpitB_r2_preflight", "scmpitB_r2_s3")
    s = F.load("scmpitB_r2_bias")
    t = domains.transfer_table(s, axis.invols_interpolator(F.load("scmpitB_r2_preflight")), 1.0)
    ref = F.load("scmpitB_r2_s3")["rows"]
    assert np.allclose(t.E, [r["E"] for r in ref], rtol=1e-9)
    assert np.allclose(t.d33_qs, [r["d33_qs_pm_per_V"] for r in ref], rtol=1e-9)
    assert abs(np.median(t.d33_qs[:-1]) - 8.81) < 0.01


def test_phase_convention_model():
    _need("scmpitA_gridB")
    from fmmpaper.io import phase_slope_through_peak
    g = F.load("scmpitA_gridB")
    assert phase_slope_through_peak(g.freq_Hz, g.Z[0, 150], g.probe.bands_Hz["CR1"]) < 0


def test_jointeb_beam_modes_stable():
    """Cancellation-free clamped-free modes: unit norm, tip value +-2, orthogonal, to mode 16."""
    import numpy as np
    from fmmpaper import jointeb
    xi = np.linspace(0, 1, 801)
    _, phi, _ = jointeb.beam_modes(xi, 16)
    trap = getattr(np, "trapezoid", None) or np.trapz
    G = np.array([[trap(phi[i] * phi[j], xi) for j in range(16)] for i in range(16)])
    assert np.allclose(np.diag(G), 1, atol=2e-4)
    assert np.abs(G - np.diag(np.diag(G))).max() < 5e-4
    assert np.allclose(np.abs(phi[:, -1]), 2, atol=1e-6)
