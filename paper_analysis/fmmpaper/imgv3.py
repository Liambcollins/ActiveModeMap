"""Signed, domain-CPD-corrected PFM images for the v3 figures (SCM-PIT R2, position A).

The two-domain frames are corrected pixel by pixel (images.cpd_correct: p = s*[z - b(V - V_s)]) and shown
signed: domain 1 positive, domain 2 negative, projected on the frame's own two-domain axis, so a
diverging colour map shows the two polarizations as red and blue with the wall at zero.
"""
from __future__ import annotations
import json
import numpy as np, pandas as pd
from . import config, images, domains

R2 = "DomainsB_SCMPIT_R2"


def _survey():
    t = pd.DataFrame(json.load(open(config.RESULTS_DIR / "fig6_main.json"))["values"]["runs"]["R2"]["table"])
    cs = domains.cpd_split(t, mask=np.ones(len(t), bool))
    c = lambda a, b: t[a].values + 1j * t[b].values
    return t, cs, c("bq_re", "bq_im") / c("Pq_re", "Pq_im"), c("bc_re", "bc_im") / c("Pc_re", "Pc_im")


def signed(z, m1, m2, V, Vbar, r, delta):
    p = images.cpd_correct(z, m1, m2, V, Vbar - delta, Vbar + delta, r, delta)
    s = np.where(m1, p, -p)
    ph = np.angle(p[images.interior(m1)].mean())
    return np.real(s * np.exp(-1j * ph))


def position_A(E_cr1):
    """QS and CR1/E images at A for 1 V and 30 mV drive, signed and CPD-corrected (pm/V)."""
    t, cs, rq, rc = _survey(); delta = cs["delta_V"]
    root = config.data_root() / R2
    J1 = json.load(open(root / "50_quant_imaging/image_cr1_quant.json")); J3 = json.load(open(root / "60_lowac_30mV/lowac_cr1_A.json"))
    xl = t.x_lever_um.values
    X = float(pd.DataFrame(json.load(open(config.RESULTS_DIR / "fig6_main.json"))["values"]["images"]).query("run == 'R2' and pos == 'A'").lever_um.iloc[0])
    ri = lambda r: np.interp(X, xl, r.real) + 1j * np.interp(X, xl, r.imag)
    out = dict(X=X, delta=delta, d33_uniform=cs["d33_uniform"], E=E_cr1)
    for tag, J, sub, qs_n, cr0_n, crv_n, vb in (("1V", J1, "50_quant_imaging", "A_qs20k_0V", "A_cr1_0V", "A_cr1_Vcpd", J1["v_cpd_V"]),
                                                ("30mV", J3, "60_lowac_30mV", "A_qs20k_0V_30mV", "A_cr1_0V_30mV", "A_cr1_Vcpdtrue_30mV", J3["v_cpd_true_V"])):
        ims = {im["name"]: im for im in J["images"]}
        rd = lambda n: images.read(root / sub / ims[n]["path"].split("\\\\")[-1].split(chr(92))[-1], ims[n].get("drive_V", J.get("drive_V", 1.0)))
        z0 = rd(cr0_n); m1, m2 = images.domain_masks(z0)
        q = signed(rd(qs_n), m1, m2, ims[qs_n]["bias_V"], vb, ri(rq), delta)
        c = signed(rd(crv_n), m1, m2, ims[crv_n]["bias_V"], vb, ri(rc), delta) / E_cr1
        e1, e2 = images.interior(m1), images.interior(m2)
        # Igor frames are (points, lines): transpose so rows are scan lines (y), as displayed on the instrument
        out[tag] = dict(qs=q.T, cr=c.T, m1=m1.T, e1=e1, e2=e2,
                        qs_med=(np.median(q[e1]), -np.median(q[e2])), cr_med=(np.median(c[e1]), -np.median(c[e2])),
                        qs_noise=np.mean([np.std(q[e1]), np.std(q[e2])]), cr_noise=np.mean([np.std(c[e1]), np.std(c[e2])]))
    return out
