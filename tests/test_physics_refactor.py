"""Regression + feature tests for the 2026-09-17 physics-path changes.

Runs anywhere (virtual instrument only, no data files, no Windows deps):

    python tests/test_physics_refactor.py

Covers
  * 5-parameter theta is unchanged (ZETA_DEFAULT reproduces the old response)
  * legacy two-domain PhysicsPosterior still recovers theta_true
  * single-domain data (no "minus") fits and feeds HybridSurrogate
  * fit_zeta recovers an injected damping; fit_geometry="setback" recovers an
    injected setback; analytic_gain recovers an injected complex gain
  * with_geometry keeps the Hz grid and does not round its cache key
  * a conjugated (wrong-convention) dataset is caught by check_phase_convention
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from activemodemap.forward_model import EBForwardModel, ProbeGeometry
from activemodemap.virtual_afm import VirtualAFM
from activemodemap.inference import PhysicsPosterior
from activemodemap.hybrid import HybridSurrogate
from activemodemap.asylum import (tune_to_complex, check_phase_convention,
                                  phase_slope_through_peak, PHASE_SIGN)

rng = np.random.default_rng(0)
ok = True


def report(name, cond, detail=""):
    global ok
    ok &= bool(cond)
    print(f"  [{'ok' if cond else 'FAIL'}] {name} {detail}")


# 1. 5-param theta unchanged ----------------------------------------------------
m = EBForwardModel()
th5 = np.array([3.2, 2.6, 1.4, 0.1, 0.05])
r5 = m.response(th5)
r6 = m.response(np.append(th5, np.log10(m.ZETA_DEFAULT)))
report("5-param == 6-param at ZETA_DEFAULT", np.allclose(r5["piezo"], r6["piezo"]))

# 2. legacy two-domain fit -------------------------------------------------------
afm = VirtualAFM(m, th5)
data = [afm.measure(x) for x in (0.35, 0.70, 0.95, 0.55, 0.85)]
post = PhysicsPosterior(m, sigma=afm.sigma, rng=rng)
th = post.fit(data)
report("legacy two-domain theta recovered", np.allclose(th, th5, atol=0.05), np.round(th, 2))
sur = HybridSurrogate(post).update(data)
plus, minus = sur.corrected_maps()
report("legacy hybrid maps", plus.shape == minus.shape == r5["piezo"].shape)

# 3. single-domain data -----------------------------------------------------------
data1 = [{"x": d["x"], "plus": d["plus"]} for d in data]
post1 = PhysicsPosterior(m, sigma=afm.sigma, rng=rng)
th1 = post1.fit(data1)
report("single-domain fit runs", np.isfinite(th1).all(), np.round(th1, 2))
sur1 = HybridSurrogate(post1).update(data1)
p1, m1 = sur1.corrected_maps()
report("single-domain hybrid: plus corrected, minus untouched",
       np.allclose(m1, post1.predict_map(domain_sign=-1.0)))

# 4. fit_zeta recovers injected damping ------------------------------------------
zeta_true = 0.006
th6 = np.append(th5, np.log10(zeta_true))
afm6 = VirtualAFM(m, th6)
data6 = [afm6.measure(x) for x in (0.35, 0.70, 0.95, 0.55, 0.85, 0.45)]
post6 = PhysicsPosterior(m, sigma=afm6.sigma, rng=rng, fit_zeta=True)
th6f = post6.fit(data6)
report("fit_zeta recovers zeta", abs(10 ** th6f[5] - zeta_true) / zeta_true < 0.25,
       f"true {zeta_true:.4f} fit {10 ** th6f[5]:.4f}")

# 5. fit_geometry='setback' recovers injected setback ----------------------------
geom_true = ProbeGeometry(tip_setback_um=8.0)
m_true = EBForwardModel(geom=geom_true)
afm_g = VirtualAFM(m_true, th5)
data_g = [afm_g.measure(x) for x in (0.35, 0.70, 0.95, 0.55, 0.85, 0.99)]
m_nom = EBForwardModel(geom=ProbeGeometry(tip_setback_um=11.0))      # wrong nominal
post_g = PhysicsPosterior(m_nom, sigma=afm_g.sigma, rng=rng, fit_geometry="setback",
                          setback_bounds_um=(3.0, 20.0))
th_g = post_g.fit(data_g)
report("fit_geometry='setback' recovers setback", abs(th_g[-1] - 8.0) < 0.6,
       f"true 8.0 fit {th_g[-1]:.2f} (nominal was 11.0)")
report("post.model carries the fitted geometry",
       abs(post_g.model.geom.tip_setback_um - th_g[-1]) < 1e-9)

# 6. analytic_gain recovers an injected complex gain -----------------------------
g_true = 0.37 * np.exp(1j * np.deg2rad(40.0))
data_gain = [{"x": d["x"], "plus": g_true * d["plus"], "minus": g_true * d["minus"]}
             for d in data]
post_gain = PhysicsPosterior(m, sigma=afm.sigma * abs(g_true), rng=rng, analytic_gain=True)
post_gain.fit(data_gain)
g_expect = g_true * 10 ** th5[4]           # A0 is pinned to 1, so the gain absorbs A0 too
report("analytic_gain recovers gain", abs(post_gain.gain / g_expect - 1) < 0.1,
       f"|g| {abs(post_gain.gain):.3f} ang {np.degrees(np.angle(post_gain.gain)):+.1f} deg")

# 7. with_geometry: same Hz grid, exact cache key ----------------------------------
m2 = m.with_geometry(f0_hz=80e3, tip_setback_um=8.0)
report("with_geometry keeps the Hz grid", np.allclose(m.f_hz, m2.f_hz))
report("with_geometry cache key is exact",
       m.with_geometry(80e3, 8.0 + 1e-9) is not m2)

# 8. phase convention check catches a conjugated dataset ---------------------------
# synthesise a Lorentzian with the MODEL's convention (phase decreasing), as if
# read from Igor with the wrong sign, and confirm the check fires
f = np.linspace(300e3, 400e3, 1001); f0, Q = 350e3, 120.0
z = 1.0 / (1 - (f / f0) ** 2 + 1j * f / (f0 * Q))            # phase decreases through peak
td_model_conv = dict(frequency=f, amplitude=np.abs(z), phase=np.degrees(np.angle(z)))
slope = phase_slope_through_peak(f, np.abs(z), np.degrees(np.angle(z)))
report("phase_slope sign detection", slope < -90, f"{slope:+.0f} deg")
import io, contextlib
buf = io.StringIO()
with contextlib.redirect_stdout(buf):
    tune_to_complex(td_model_conv, check_convention=True)        # default PHASE_SIGN=-1
report("check_convention warns when raw phase already has the model's sign",
       "PHASE CONVENTION" in buf.getvalue())
buf = io.StringIO()
td_igor_conv = dict(td_model_conv, phase=-td_model_conv["phase"])   # what the Cypher gives
with contextlib.redirect_stdout(buf):
    _, Z = tune_to_complex(td_igor_conv, check_convention=True)
report("no warning for Cypher-convention data", "PHASE CONVENTION" not in buf.getvalue())
report("tune_to_complex output has the model's rotation",
       phase_slope_through_peak(f, np.abs(Z), np.degrees(np.angle(Z))) < -90)

print("\nALL OK" if ok else "\nSOME TESTS FAILED")
sys.exit(0 if ok else 1)
