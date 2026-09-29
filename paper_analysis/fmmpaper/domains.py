"""Two-domain bias decomposition, contact potential, enhancement and d33.

Model, per position and frequency, on antiparallel domains s = 1, 2:

    Z_s(V) = a_s + b_s V        (complex least squares over the bias points)
    P      = (a_1 - a_2) / 2    piezoresponse: flips sign between domains
    b      = (b_1 + b_2) / 2    electrostatic slope: same on both domains
    V_cpd  = Re[-(a_1 + a_2) / (2 b)]

Checks that the model holds: flip = arg(a_1 / a_2) should be ~180 deg and
|b_2/b_1| ~ 1. Note flip uses the 0 V intercepts (archived convention), so a large
electrostatic term at 0 V (V_cpd far from 0, or large |b|/|P|) pulls it off 180 deg
even when the model is exact. For R2 CR1 (|b|/|P| ~ 0.19, V_cpd ~ 1 V) this alone gives ~173 deg,
part of why the measured CR1 flip sits at
150-170 deg while the QS flip stays near 182 deg. Bias must be interleaved in acquisition; a monotonic sweep
(R2 stage 41) fails these checks at CR1.

This reproduces r2an/common.decompose and r2an/s3_transfer.py exactly, but is
vectorised over frequency so the same fit yields P(x, f), b(x, f) maps.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .io import Series
from .spectra import peak


def _linfit(V, Z):
    """Complex LS fit of Z[n_bias, ...] = a + b V. Returns a, b, rms residual fraction."""
    V = np.asarray(V, float)
    A = np.vstack([np.ones_like(V), V]).T                       # (n, 2)
    shp = Z.shape[1:]
    coef, *_ = np.linalg.lstsq(A, Z.reshape(len(V), -1), rcond=None)
    a, b = coef[0].reshape(shp), coef[1].reshape(shp)
    res = Z - (a[None] + b[None] * V.reshape(-1, *[1] * len(shp)))
    frac = np.sqrt(np.mean(np.abs(res) ** 2, 0)) / np.maximum(np.sqrt(np.mean(np.abs(Z) ** 2, 0)), 1e-30)
    return a, b, frac


def decompose(V1, Z1, V2, Z2) -> dict:
    """Two-domain decomposition. Z1, Z2: [n_bias, ...] complex (any trailing shape)."""
    a1, b1, r1 = _linfit(V1, Z1)
    a2, b2, r2 = _linfit(V2, Z2)
    P = (a1 - a2) / 2.0
    b = (b1 + b2) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        vcpd = np.real(-(a1 + a2) / (2.0 * b))
        ratio = np.abs(b) / np.abs(P)
        bal = np.abs(b2) / np.abs(b1)
    flip = np.degrees(np.angle(a1 * np.conj(a2))) % 360.0
    return dict(a1=a1, b1=b1, a2=a2, b2=b2, P=P, b=b, v_cpd=vcpd, ratio=ratio,
                flip_deg=flip, b_balance=bal, resid1=r1, resid2=r2)


def split_spots(s: Series, spots=(1, 2)):
    i1 = s.where(spot=spots[0]); i2 = s.where(spot=spots[1])
    V1 = s.conditions.bias_V.values[i1]; V2 = s.conditions.bias_V.values[i2]
    return i1, V1, i2, V2


def decompose_series(s: Series, band=None, spots=(1, 2)) -> dict:
    """Frequency-resolved decomposition of a two-domain bias survey.

    Returns arrays shaped [position, freq] (restricted to ``band`` if given).
    """
    i1, V1, i2, V2 = split_spots(s, spots)
    m = s.band(band) if band is not None else np.ones(s.freq_Hz.size, bool)
    out = decompose(V1, s.Z[i1][:, :, m], V2, s.Z[i2][:, :, m])
    out["freq_Hz"] = s.freq_Hz[m]
    out["x_um"] = s.x_um
    return out


def transfer_table(s: Series, invols_fn, vac_V: float, cr_band="CR1", spots=(1, 2),
                   ref_bias: float = 0.0) -> pd.DataFrame:
    """Per-position channel table: the core of the 'E is not Q' result.

    For each position:
      * QS channel: complex mean over the quasi-static band, per condition -> decompose
      * CR channel: complex value at the CR peak of the (ref_bias, spot 1) spectrum,
        taken at that same frequency for every condition -> decompose
      * E = |P_CR| / |P_QS| (in-situ enhancement), Q from the -3 dB width
      * d33_qs = |P_QS| * InvOLS / amp_divisor / Vac  (pm/V)
    Mirrors r2an/s3_transfer.py.
    """
    i1, V1, i2, V2 = split_spots(s, spots)
    iref = int(np.intersect1d(s.where(bias_V=ref_bias), i1)[0])
    qm = s.band("QS"); band = s.probe.bands_Hz[cr_band]
    rows = []
    for j, x in enumerate(s.x_um):
        zq = s.Z[:, j, qm].mean(-1)
        dq = decompose(V1, zq[i1], V2, zq[i2])
        pk = peak(s.freq_Hz, s.Z[iref, j], band)
        zc = s.Z[:, j, pk["k"]]
        dc = decompose(V1, zc[i1], V2, zc[i2])
        inv = float(invols_fn(x))
        conv = inv / s.probe.amp_divisor / vac_V * 1e12       # raw V -> pm/V
        rows.append(dict(
            x_um=float(x), x_clamp_um=float(x - s.probe.clamp_offset_um), invols=inv,
            f_cr_Hz=pk["f_Hz"], Q=pk["Q"],
            P_qs=float(abs(dq["P"])), P_cr=float(abs(dc["P"])),
            E=float(abs(dc["P"]) / abs(dq["P"])),
            vcpd_qs=float(dq["v_cpd"]), vcpd_cr=float(dc["v_cpd"]),
            ratio_qs=float(dq["ratio"]), ratio_cr=float(dc["ratio"]),
            flip_qs=float(dq["flip_deg"]), flip_cr=float(dc["flip_deg"]),
            bal_cr=float(dc["b_balance"]),
            d33_qs=float(abs(dq["P"]) * conv),
            d33_cr_over_Q=float(abs(dc["P"]) * conv / pk["Q"]),
            # complex P and b in pm/V (b: pm/V per V of DC bias), for cpd_split
            Pq_re=float((dq["P"] * conv).real), Pq_im=float((dq["P"] * conv).imag),
            bq_re=float((dq["b"] * conv).real), bq_im=float((dq["b"] * conv).imag),
            Pc_re=float((dc["P"] * conv).real), Pc_im=float((dc["P"] * conv).imag),
            bc_re=float((dc["b"] * conv).real), bc_im=float((dc["b"] * conv).imag),
        ))
    return pd.DataFrame(rows)


def cpd_split(table: pd.DataFrame, mask=None) -> dict:
    """Remove a domain-dependent contact-potential term from the two-domain piezoresponse.

    If the two domains have different contact potentials V1, V2, then
        Z_s(V) = s*p + b(x) (V - V_s)   and   P = (a1 - a2)/2 = p + delta * b(x),
    with delta = -(V1 - V2)/2. The surviving term has the electrostatic spatial shape b(x),
    which crosses zero only at the electrostatic blind spot. P therefore varies along the
    lever even when d33 does not.

    delta is fitted in the QUASI-STATIC channel, assuming d33 is uniform along the lever:
        P_qs(x) = D + delta * b_qs(x)      (D complex, delta real; linear least squares).
    At resonance b_cr is proportional to P_cr (both drive the same mode), so the CR1 channel
    cannot constrain delta on its own; the QS value is applied to both channels.
    Returns delta_V, the uniform QS d33 |D|, and corrected |P| arrays (pm/V) for both channels.
    """
    t = table
    m = np.ones(len(t), bool) if mask is None else np.asarray(mask, bool)
    Pq = t.Pq_re.values + 1j * t.Pq_im.values; bq = t.bq_re.values + 1j * t.bq_im.values
    Pc = t.Pc_re.values + 1j * t.Pc_im.values; bc = t.bc_re.values + 1j * t.bc_im.values
    n = int(m.sum())
    A = np.block([[np.ones((n, 1)), np.zeros((n, 1)), bq[m].real[:, None]],
                  [np.zeros((n, 1)), np.ones((n, 1)), bq[m].imag[:, None]]])
    y = np.concatenate([Pq[m].real, Pq[m].imag])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    D, delta = sol[0] + 1j * sol[1], float(sol[2])
    Pq_c, Pc_c = Pq - delta * bq, Pc - delta * bc
    return dict(delta_V=delta, dV_domains_V=-2 * delta, d33_uniform=float(abs(D)),
                d33_qs_corr=np.abs(Pq_c), P_cr_corr=np.abs(Pc_c), E_corr=np.abs(Pc_c) / np.abs(Pq_c))


def d33_from_enhancement(table: pd.DataFrame, E_model, vac_V: float = 1.0,
                         amp_divisor: float = 32.0) -> np.ndarray:
    """d33 (pm/V) from the CR channel, given a model or predicted enhancement per position."""
    return (table.P_cr.values * table.invols.values / amp_divisor / vac_V * 1e12
            / np.asarray(E_model, float))


def v0_null_bias(P, b, v_cpd):
    """Bias minimising |P + b (V - V_cpd)| on one domain: V_cpd - Re(P b*)/|b|^2."""
    return v_cpd - np.real(P * np.conj(b)) / np.abs(b) ** 2
