"""Blind Euler-Bernoulli fit of a contact state (port of r2an fit_eb_r2.py, extended).

The residual sees only
  * the contact-resonance frequencies (CR1-CR3),
  * the CR1 quality factor,
  * the SHAPE of 1/InvOLS(x) (gain projected out), and optionally
  * interior node positions of CR2 / CR3 in the lever frame (distance from clamp).
No amplitude and no enhancement enters, so E(x) = |P_CR1|/|P_QS| stays a blind test.

Positions must be in the LEVER frame (x - x0(t)); see calib.frame_anchors / x0_at.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import least_squares

from . import config

config.ensure_activemodemap()
from activemodemap.forward_model import EBForwardModel, ProbeGeometry  # noqa: E402

PARAMS = ("log_alpha", "log_kcone", "setback_um", "log_zeta", "tip_height_um")
LO = np.array([1.5, 0.0, 0.3, -3.5, 5.0])
HI = np.array([5.0, 4.0, 40.0, -1.0, 25.0])


@dataclass
class ContactState:
    name: str
    cr_Hz: np.ndarray                  # measured CR1, CR2, CR3
    Q1: float                          # measured CR1 quality factor
    x_static_um: np.ndarray            # lever-frame positions of the static-shape points
    invols_static: np.ndarray          # InvOLS at those positions
    nodes_um: dict = field(default_factory=dict)   # {"CR2": [..], "CR3": [..]} lever frame
    f0_free_Hz: float = 63.801e3
    k_lever: float = 1.697
    L_um: float = 226.5


def _model(p, st: ContactState, f_list):
    la, lk, sb, lz, th = p
    g = ProbeGeometry(name="SCM-PIT", f0_hz=st.f0_free_Hz, k_lever=st.k_lever, L_um=st.L_um,
                      tip_setback_um=sb, tip_height_um=th, tilt_deg=11.0)
    m = EBForwardModel(geom=g, n_modes=14, nx=453, nf=2)
    m.omega = np.asarray(f_list, float) / m.freq_scale_hz
    m.f_hz = m.omega * m.freq_scale_hz
    return m, m.response(np.array([la, lk, 2.3, 0.0, 0.0, lz]))


def peaks(p, st, f_lo=150e3, f_hi=2.1e6, n=2500):
    m, r = _model(p, st, np.linspace(f_lo, f_hi, n))
    i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r["piezo"][:, i])
    pk = sorted([k for k in range(1, a.size - 1) if a[k] >= a[k - 1] and a[k] >= a[k + 1]],
                key=lambda k: -a[k])
    out = []
    for k in pk:
        if all(abs(m.f_hz[k] - m.f_hz[o]) > 60e3 for o in out):
            out.append(k)
        if len(out) == 3:
            break
    return np.sort(m.f_hz[out]) if len(out) == 3 else None


def model_nodes(p, st, f_mode):
    """Interior amplitude minima of the piezo response at f_mode, in um from the clamp."""
    m, r = _model(p, st, [f_mode])
    a = np.abs(r["piezo"][0]); xu = m.xi * st.L_um
    tip = st.L_um - p[2]
    idx = [i for i in range(2, a.size - 2) if a[i] <= a[i - 1] and a[i] <= a[i + 1]
           and a[i] < 0.3 * a.max() and xu[i] < tip - 3]
    return [float(xu[i]) for i in idx]


def q_model(p, st, f1):
    m, r = _model(p, st, np.linspace(f1 - 10e3, f1 + 10e3, 1201))
    i = np.argmin(np.abs(m.xi - 0.55)); a = np.abs(r["piezo"][:, i])
    j = int(np.argmax(a)); h = a[j] / np.sqrt(2); lo = hi = j
    while lo > 0 and a[lo] > h:
        lo -= 1
    while hi < a.size - 1 and a[hi] > h:
        hi += 1
    return m.f_hz[j] / (m.f_hz[hi] - m.f_hz[lo])


def static_shape(p, st, x_um):
    m, r = _model(p, st, [30e3])
    return np.interp(x_um, m.xi * st.L_um, np.abs(r["piezo"][0]))


def residual(p, st: ContactState, use_nodes=True, node_scale_um=100.0, sig_f=None, sig_static=None, sig_Q=None):
    """Default weights reproduce the archived fit (freq x3, static x1, Q x1 on log residuals).
    Passing sig_f / sig_static / sig_Q (fractional 1-sigma) weights each block by 1/sigma instead:
    use sig_f ~ 0.015 (the shared-geometry EB frequency floor, Phase 2g), sig_static ~ 0.02
    (InvOLS repeatability), sig_Q ~ 0.05."""
    wf, ws, wq = (3.0, 1.0, 1.0) if sig_f is None else (1 / sig_f, 1 / sig_static, 1 / sig_Q)
    fk = peaks(p, st)
    n_nodes = sum(len(v) for v in st.nodes_um.values()) if use_nodes else 0
    if fk is None:
        return np.full(3 + st.x_static_um.size + 1 + n_nodes, 5.0 * max(wf, ws, wq) / 3)
    zq = static_shape(p, st, st.x_static_um)
    y = 1.0 / st.invols_static
    g = np.sum(y * zq) / np.sum(zq * zq)
    r_static = ws * np.log(y / (g * zq))
    r_Q = wq * np.log(q_model(p, st, fk[0]) / st.Q1)
    r = [wf * np.log(fk / st.cr_Hz), r_static, [r_Q]]
    if use_nodes:
        for k, mode in ((1, "CR2"), (2, "CR3")):
            meas = st.nodes_um.get(mode, [])
            if not meas:
                continue
            mod = model_nodes(p, st, fk[k])
            for xm in meas:
                d = min((abs(xm - xx) for xx in mod), default=50.0)
                r.append([d / node_scale_um])
    return np.concatenate([np.atleast_1d(v) for v in r])


def fit(st: ContactState, use_nodes=True, starts=None, max_nfev=80, verbose=False, lo=None, hi=None, **wkw):
    lo = LO if lo is None else np.asarray(lo, float); hi = HI if hi is None else np.asarray(hi, float)
    starts = starts or [[la, lk, 5.0, np.log10(2e-3), th]
                        for la in (2.4, 2.8, 3.2) for lk in (1.0, 2.0, 3.0) for th in (10.0, 15.0)]
    best = None
    for p0 in starts:
        try:
            r = least_squares(residual, p0, args=(st, use_nodes), kwargs=wkw, bounds=(lo, hi),
                              diff_step=2e-3, max_nfev=max_nfev)
        except Exception:
            continue
        if best is None or r.cost < best.cost:
            best = r
            if verbose:
                print(f"  cost {r.cost:.5f} from {np.round(p0, 2)}")
    return best


def summarize(p, st: ContactState) -> dict:
    fk = peaks(p, st)
    out = dict(state=st.name, k_ratio=10 ** p[0], kcone_ratio=10 ** p[1], setback_um=p[2],
               zeta=10 ** p[3], tip_height_um=p[4],
               cr_model_kHz=(fk / 1e3).round(2).tolist() if fk is not None else None,
               cr_err_pct=(100 * (fk / st.cr_Hz - 1)).round(2).tolist() if fk is not None else None,
               Q_model=q_model(p, st, fk[0]) if fk is not None else None, Q_meas=st.Q1)
    if fk is not None:
        out["nodes_model_um"] = {"CR2": model_nodes(p, st, fk[1]), "CR3": model_nodes(p, st, fk[2])}
        out["nodes_meas_um"] = st.nodes_um
    zq = static_shape(p, st, st.x_static_um); y = 1 / st.invols_static
    g = np.sum(y * zq) / np.sum(zq * zq)
    out["static_rms_pct"] = 100 * float(np.sqrt(np.mean(np.log(y / (g * zq)) ** 2)))
    return out


def enhancement(p, st: ContactState, x_um, f_qs_Hz=30e3):
    """Blind EB enhancement E(x) = |z_piezo(f_CR1, x)| / |z_piezo(f_qs, x)| in the lever frame.

    Same definition as the archived fit_eb_r2.py (E_mod = zc / zq). Also returns the model's
    quasi-static and CR1 shapes so d33 can be formed from either channel.
    """
    fk = peaks(p, st)
    if fk is None:
        raise ValueError("model has no three contact resonances for these parameters")
    m, r = _model(p, st, [f_qs_Hz, fk[0]])
    xu = m.xi * st.L_um
    zq = np.interp(x_um, xu, np.abs(r["piezo"][0]))
    zc = np.interp(x_um, xu, np.abs(r["piezo"][1]))
    return zc / zq, zq, zc, float(fk[0])
