"""Map-route analysis of a two-domain sparse survey (numpy only).

Per laser position x and frequency f, on antiparallel domains s = 1, 2:

    Z_s(V) = a_s + b_s V            complex least squares over the DC biases
    P_raw  = (a_1 - a_2) / 2        piezoresponse (flips sign between domains)
    b      = (b_1 + b_2) / 2        electrostatic slope (same on both domains)
    P      = P_raw - delta * b      domain-potential correction; delta fitted in the
                                    quasi-static band assuming d_eff is uniform along the lever
    E_map(x) = |P_CR1(x)| / |P_QS(x)|          resonance enhancement, map route
    d_eff,QS  = |P_QS| * InvOLS / G / V_ac      (pm/V)
    d_eff,CR1 = |P_CR1| * InvOLS / G / V_ac / E

E_map is interpolated along the lever by fitting the corrected P maps (CR1 band
and QS band) with a low-order Chebyshev basis in x — the data-driven route of the
paper — and validated by leaving each measured position out in turn.

These are the same conventions as `paper_analysis/fmmpaper/domains.py`
(`decompose`, `transfer_table(track=True)`, `cpd_split`), re-implemented here
without pandas/scipy so they run inside the instrument kernel.
"""
from __future__ import annotations

import numpy as np

from ..lowrank import chebyshev_basis, reconstruct_map, band_mask


# ----------------------------------------------------------------------------------- #
#  small spectral helpers                                                              #
# ----------------------------------------------------------------------------------- #
def _parab(f, a, k):
    if 0 < k < a.size - 1:
        y0, y1, y2 = a[k - 1], a[k], a[k + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(f[k] + 0.5 * (y0 - y2) / den * (f[k + 1] - f[k]))
    return float(f[k])


def _q_factor(fb, a, k):
    half = a[k] / np.sqrt(2); lo = k; hi = k
    while lo > 0 and a[lo] > half: lo -= 1
    while hi < a.size - 1 and a[hi] > half: hi += 1
    if lo == 0 or hi == a.size - 1 or hi <= lo:
        return float("nan")
    fl = np.interp(half, [a[lo], a[lo + 1]], [fb[lo], fb[lo + 1]])
    fh = np.interp(half, [a[hi], a[hi - 1]], [fb[hi], fb[hi - 1]])
    return float(_parab(fb, a, k) / (fh - fl)) if fh > fl else float("nan")


def peak(freq, z, band):
    """Largest |z| inside band: dict(f_Hz, amp, z, k [index into freq], Q, fwhm_Hz)."""
    m = band_mask(freq, band)
    fb, zb = freq[m], z[..., m]; a = np.abs(zb)
    k = int(np.argmax(a))
    Q = _q_factor(fb, a, k)
    f0 = _parab(fb, a, k)
    return dict(f_Hz=f0, amp=float(a[k]), z=complex(zb[k]), k=int(np.flatnonzero(m)[k]), Q=Q,
                fwhm_Hz=float(f0 / Q) if np.isfinite(Q) and Q > 0 else float("nan"))


def _linfit(V, Z):
    """Complex LS fit Z[n_bias, ...] = a + b V. Returns a, b."""
    V = np.asarray(V, float)
    A = np.stack([np.ones_like(V), V], 1)
    shp = Z.shape[1:]
    coef = np.linalg.lstsq(A, Z.reshape(len(V), -1), rcond=None)[0]
    return coef[0].reshape(shp), coef[1].reshape(shp)


def decompose(V1, Z1, V2, Z2):
    """Two-domain decomposition of Z[n_bias, ...] on spots 1 and 2 (any trailing shape)."""
    a1, b1 = _linfit(V1, Z1); a2, b2 = _linfit(V2, Z2)
    P = (a1 - a2) / 2.0; b = (b1 + b2) / 2.0
    with np.errstate(divide="ignore", invalid="ignore"):
        v_cpd = np.real(-(a1 + a2) / (2.0 * b))
        ratio = np.abs(b) / np.abs(P)
        bal = np.abs(b2) / np.abs(b1)
    flip = np.degrees(np.angle(a1 * np.conj(a2))) % 360.0
    return dict(P=P, b=b, v_cpd=v_cpd, ratio=ratio, flip_deg=flip, b_balance=bal, a1=a1, a2=a2, b1=b1, b2=b2)


def _cond_attr(c, name):
    return c[name] if isinstance(c, dict) else getattr(c, name)


def split_spots(series, spots):
    conds = series["conditions"]
    i1 = [i for i, c in enumerate(conds) if _cond_attr(c, "spot") == spots[0]]
    i2 = [i for i, c in enumerate(conds) if _cond_attr(c, "spot") == spots[1]]
    if len(i1) < 3 or len(i2) < 3:
        raise ValueError(f"need >= 3 biases on each spot {spots}; found {len(i1)} and {len(i2)}")
    V1 = np.array([_cond_attr(conds[i], "bias_V") for i in i1], float)
    V2 = np.array([_cond_attr(conds[i], "bias_V") for i in i2], float)
    return i1, V1, i2, V2


def find_cr1(series, prior_band_Hz, ref_cond=0):
    """CR1 frequency as the median over positions of the reference condition's peak in the prior band."""
    F, Z = series["freq_Hz"], series["Z"]
    fs = [peak(F, Z[ref_cond, j], prior_band_Hz)["f_Hz"] for j in range(Z.shape[1])
          if np.isfinite(Z[ref_cond, j]).all()]
    return float(np.median(fs))


# ----------------------------------------------------------------------------------- #
#  per-position channel table                                                          #
# ----------------------------------------------------------------------------------- #
def channel_table(series, spots, qs_band_Hz, cr_band_Hz, track_peaks=True, ref_bias_V=0.0,
                  invols_by_x=None):
    """QS and CR1 channels per laser position (dict of arrays, one entry per position).

    QS channel : complex mean over `qs_band_Hz` per condition -> two-domain decomposition.
    CR channel : complex value at the CR1 peak. `track_peaks=True` reads every condition at
                 its OWN peak (resonance moves with bias and domain); False reads all at the
                 (ref_bias, spot 1) peak bin. Q is the median over conditions (tracked) or the
                 reference condition's Q.
    """
    i1, V1, i2, V2 = split_spots(series, spots)
    conds = series["conditions"]
    iref = [i for i in i1 if abs(_cond_attr(conds[i], "bias_V") - ref_bias_V) < 1e-9]
    iref = iref[0] if iref else i1[0]
    F, Z, xs = series["freq_Hz"], series["Z"], np.asarray(series["x_um"], float)
    qm = band_mask(F, qs_band_Hz)
    inv_tab = {round(float(k), 1): float(v) for k, v in (invols_by_x or {}).items()}
    cols = {k: [] for k in ("x_um", "invols", "invols_known", "f_cr_Hz", "Q", "fwhm_Hz", "f_cr_spread_Hz",
                            "P_qs", "b_qs", "P_cr", "b_cr", "v_cpd_qs", "v_cpd_cr", "ratio_qs", "ratio_cr",
                            "flip_qs", "flip_cr", "b_balance_cr", "complete")}
    for j, x in enumerate(xs):
        ok = np.isfinite(Z[:, j]).all()
        cols["complete"].append(bool(ok)); cols["x_um"].append(float(x))
        inv = inv_tab.get(round(float(x), 1))
        cols["invols_known"].append(inv is not None)
        cols["invols"].append(inv if inv is not None else np.nan)
        if not ok:
            for k in cols:
                if k not in ("x_um", "invols", "invols_known", "complete"):
                    cols[k].append(np.nan)
            continue
        zq = Z[:, j, qm].mean(-1)
        dq = decompose(V1, zq[i1], V2, zq[i2])
        pk = peak(F, Z[iref, j], cr_band_Hz)
        if track_peaks:
            pks = [peak(F, Z[c, j], cr_band_Hz) for c in range(Z.shape[0])]
            zc = np.array([q["z"] for q in pks])
            Q = float(np.nanmedian([q["Q"] for q in pks])); spread = float(np.ptp([q["f_Hz"] for q in pks]))
        else:
            zc = Z[:, j, pk["k"]]; Q = pk["Q"]; spread = 0.0
        dc = decompose(V1, zc[i1], V2, zc[i2])
        cols["f_cr_Hz"].append(pk["f_Hz"]); cols["Q"].append(Q); cols["fwhm_Hz"].append(pk["fwhm_Hz"])
        cols["f_cr_spread_Hz"].append(spread)
        cols["P_qs"].append(complex(dq["P"])); cols["b_qs"].append(complex(dq["b"]))
        cols["P_cr"].append(complex(dc["P"])); cols["b_cr"].append(complex(dc["b"]))
        cols["v_cpd_qs"].append(float(dq["v_cpd"])); cols["v_cpd_cr"].append(float(dc["v_cpd"]))
        cols["ratio_qs"].append(float(dq["ratio"])); cols["ratio_cr"].append(float(dc["ratio"]))
        cols["flip_qs"].append(float(dq["flip_deg"])); cols["flip_cr"].append(float(dc["flip_deg"]))
        cols["b_balance_cr"].append(float(dc["b_balance"]))
    out = {k: np.asarray(v) for k, v in cols.items()}
    out["E_raw"] = np.abs(out["P_cr"]) / np.abs(out["P_qs"])
    return out


def fit_delta(table, mask=None):
    """Domain-potential term: P_qs(x) = D + delta * b_qs(x), D complex, delta real (LSQ over positions).

    The uniformity assumption is on d_eff, i.e. on the DISPLACEMENT, so the fit is done on
    P_qs * InvOLS(x) (InvOLS changes by an order of magnitude along the lever). Positions
    without a known InvOLS are left out; with fewer than 3 known, the raw volts are used
    and the result flagged (`on_displacement=False`).
    Returns dict(delta_V, dV_domains_V, D [uniform QS response, same units as the fit],
    rms_resid_frac, n_used, on_displacement).
    """
    ok = table["complete"] & np.isfinite(np.abs(table["P_qs"]))
    m = ok if mask is None else (ok & np.asarray(mask, bool))
    inv = np.asarray(table["invols"], float)
    known = m & np.isfinite(inv)
    on_disp = int(known.sum()) >= 3
    if on_disp:
        m = known; scale = inv[m]
    else:
        scale = np.ones(int(m.sum()))
    Pq, bq = table["P_qs"][m] * scale, table["b_qs"][m] * scale
    n = int(m.sum())
    if n < 3:
        raise ValueError("need >= 3 complete positions to fit delta")
    A = np.block([[np.ones((n, 1)), np.zeros((n, 1)), bq.real[:, None]],
                  [np.zeros((n, 1)), np.ones((n, 1)), bq.imag[:, None]]])
    y = np.concatenate([Pq.real, Pq.imag])
    sol, *_ = np.linalg.lstsq(A, y, rcond=None)
    D, delta = sol[0] + 1j * sol[1], float(sol[2])
    fit = D + delta * bq
    resid = float(np.sqrt(np.mean(np.abs(Pq - fit) ** 2)) / np.sqrt(np.mean(np.abs(Pq) ** 2)))
    return dict(delta_V=delta, dV_domains_V=-2 * delta, D=D, rms_resid_frac=resid, n_used=n,
                on_displacement=on_disp)


def apply_delta(table, delta_V):
    """Corrected channels and the measured enhancement per position."""
    Pq_c = table["P_qs"] - delta_V * table["b_qs"]
    Pc_c = table["P_cr"] - delta_V * table["b_cr"]
    with np.errstate(divide="ignore", invalid="ignore"):
        E = np.abs(Pc_c) / np.abs(Pq_c)
    return dict(P_qs_corr=Pq_c, P_cr_corr=Pc_c, E_meas=E)


def deff_qs_pm_per_V(P_qs_abs, invols_m_per_V, drive_V, amp_divisor=32.0):
    return np.asarray(P_qs_abs, float) * np.asarray(invols_m_per_V, float) / amp_divisor / drive_V * 1e12


def deff_cr_pm_per_V(P_cr_abs, invols_m_per_V, drive_V, E, amp_divisor=32.0):
    return deff_qs_pm_per_V(P_cr_abs, invols_m_per_V, drive_V, amp_divisor) / np.asarray(E, float)


# ----------------------------------------------------------------------------------- #
#  frequency-resolved maps and the along-lever interpolation                           #
# ----------------------------------------------------------------------------------- #
def _align_cr_peaks(series, i_all, band_Hz, iref):
    """Shift every condition's CR-band spectrum so its peak sits at the reference condition's
    peak of the same position (frequency-resolved form of peak tracking)."""
    F, Z = series["freq_Hz"], series["Z"].copy()
    m = band_mask(F, band_Hz); fb = F[m]
    for j in range(Z.shape[1]):
        if not np.isfinite(Z[:, j]).all():
            continue
        f0 = peak(F, Z[iref, j], band_Hz)["f_Hz"]
        for c in range(Z.shape[0]):
            df = f0 - peak(F, Z[c, j], band_Hz)["f_Hz"]
            if abs(df) < 1e-9:
                continue
            z = Z[c, j, m]
            Z[c, j, m] = np.interp(fb - df, fb, z.real) + 1j * np.interp(fb - df, fb, z.imag)
    return Z


def decompose_maps(series, spots, band_Hz, align=False, ref_bias_V=0.0):
    """P_raw(x, f) and b(x, f) inside `band_Hz`; returns dict with freq_Hz, P, b, x_um."""
    i1, V1, i2, V2 = split_spots(series, spots)
    F = series["freq_Hz"]; m = band_mask(F, band_Hz)
    Z = series["Z"]
    if align:
        conds = series["conditions"]
        iref = [i for i in i1 if abs(_cond_attr(conds[i], "bias_V") - ref_bias_V) < 1e-9]
        Z = _align_cr_peaks(series, i1 + i2, band_Hz, iref[0] if iref else i1[0])
    d = decompose(V1, Z[i1][:, :, m], V2, Z[i2][:, :, m])
    return dict(freq_Hz=F[m], P=d["P"], b=d["b"], x_um=np.asarray(series["x_um"], float))


def _emap_from_maps(x_grid, x_meas, Pcr, Pqs, rank, use=None):
    """Chebyshev interpolation of the corrected P maps along x; E(x) = max_f|Pcr| / mean_f|Pqs|."""
    use = np.arange(len(x_meas)) if use is None else np.asarray(use, int)
    sel = [int(np.argmin(np.abs(x_grid - x_meas[i]))) for i in use]
    r = int(min(rank, len(sel)))
    rc = reconstruct_map(x_grid, sel, Pcr[use], r)
    rq = reconstruct_map(x_grid, sel, Pqs[use], r)
    Acr = np.abs(rc["Zrec"]).max(1); Aqs = np.abs(rq["Zrec"]).mean(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        E = Acr / Aqs
        # crude 1-sigma band from the per-(x, f) prediction std at the CR peak and the QS mean
        kpk = np.abs(rc["Zrec"]).argmax(1)
        s_cr = rc["std"][np.arange(len(x_grid)), kpk]; s_qs = rq["std"].mean(1)
        E_lo = np.maximum(Acr - s_cr, 0) / (Aqs + s_qs)
        E_hi = (Acr + s_cr) / np.maximum(Aqs - s_qs, 1e-30)
    return dict(E=E, E_lo=E_lo, E_hi=E_hi, Acr=Acr, Aqs=Aqs, Pcr_rec=rc["Zrec"], Pqs_rec=rq["Zrec"], rank=r)


def emap_along_lever(series, spots, qs_band_Hz, cr_band_Hz, delta_V=0.0, rank=4, grid_step_um=1.0,
                     x_grid=None, align=True, ref_bias_V=0.0, loo=True):
    """E_map on a fine grid along the lever, from the corrected two-domain P maps.

    Returns dict(x_grid, E, E_lo, E_hi, E_at_meas, E_loo, x_meas, maps...). `E_loo[j]` is E at
    measured position j predicted from the other positions (the paper's leave-one-out test).
    """
    mq = decompose_maps(series, spots, qs_band_Hz, align=False)
    mc = decompose_maps(series, spots, cr_band_Hz, align=align, ref_bias_V=ref_bias_V)
    x_meas = mq["x_um"]
    ok = np.isfinite(mq["P"]).all(1) & np.isfinite(mc["P"]).all(1)
    Pqs = (mq["P"] - delta_V * mq["b"])[ok]; Pcr = (mc["P"] - delta_V * mc["b"])[ok]
    xm = x_meas[ok]
    if x_grid is None:
        x_grid = np.arange(np.floor(xm.min()), np.ceil(xm.max()) + 0.5 * grid_step_um, grid_step_um)
    x_grid = np.asarray(x_grid, float)
    full = _emap_from_maps(x_grid, xm, Pcr, Pqs, rank)
    E_meas_map = np.abs(Pcr).max(1) / np.abs(Pqs).mean(1)
    E_loo = np.full(xm.size, np.nan)
    if loo and xm.size >= 4:
        for j in range(xm.size):
            use = np.array([i for i in range(xm.size) if i != j])
            r = _emap_from_maps(x_grid, xm, Pcr, Pqs, min(rank, use.size - 2), use=use)
            E_loo[j] = np.interp(xm[j], x_grid, r["E"])
    return dict(x_grid=x_grid, E=full["E"], E_lo=full["E_lo"], E_hi=full["E_hi"], Acr=full["Acr"], Aqs=full["Aqs"],
                Pcr_rec=full["Pcr_rec"], Pqs_rec=full["Pqs_rec"], f_cr_Hz=mc["freq_Hz"], f_qs_Hz=mq["freq_Hz"],
                x_meas=xm, Pcr_meas=Pcr, Pqs_meas=Pqs, E_at_meas=E_meas_map, E_loo=E_loo, rank=full["rank"],
                loo_ratio=E_meas_map / E_loo)


def qs_blind_spot(table, delta_V, x_grid, rank=4):
    """Electrostatic blind spot of the quasi-static channel, from the unit-free ratio r(x) = b_QS / P_QS.

    r is the electrostatic bias slope relative to the (corrected) piezoresponse at each measured
    position (InvOLS cancels). It is interpolated along the lever with the Chebyshev basis and
    projected on the phasor of its largest value; the blind spot is the zero crossing nearest the
    tip. With no crossing inside the measured span (the usual case on a stiff lever: r falls
    monotonically toward the tip) the stop is the measured end with the smallest |r|, and the
    residual error of an UNCORRECTED QS image there is delta * |r| (a few per cent).
    Returns dict(x_esbs_um, crossing, r_abs_grid, r_abs_meas, x_meas, r_at_stop, raw_error_pct).
    """
    ok = table["complete"] & np.isfinite(np.abs(table["P_qs"]))
    xm = table["x_um"][ok]
    Pc = (table["P_qs"] - delta_V * table["b_qs"])[ok]
    r = table["b_qs"][ok] / Pc
    x_grid = np.asarray(x_grid, float)
    sel = [int(np.argmin(np.abs(x_grid - x))) for x in xm]
    k = int(min(rank, xm.size))
    B = chebyshev_basis(x_grid, k)
    rg = B @ np.linalg.lstsq(B[sel], r, rcond=None)[0]
    u = r[np.argmax(np.abs(r))]; u = u / abs(u)
    proj = (rg * np.conj(u)).real
    lo, hi = xm.min(), xm.max()
    idx = np.flatnonzero((x_grid >= lo) & (x_grid <= hi))
    cross = [i for i in idx[:-1] if proj[i] * proj[i + 1] < 0]
    if cross:
        i = cross[-1]
        x0 = float(x_grid[i] - proj[i] * (x_grid[i + 1] - x_grid[i]) / (proj[i + 1] - proj[i])); crossing = True
    else:
        j = int(np.argmin(np.abs(r))); x0 = float(xm[j]); crossing = False
    r_stop = float(np.interp(x0, x_grid, np.abs(rg))) if crossing else float(np.abs(r[j]))
    return dict(x_esbs_um=x0, crossing=crossing, r_abs_grid=np.abs(rg), r_proj_grid=proj, x_grid=x_grid,
                r_abs_meas=np.abs(r), x_meas=xm, r_at_stop=r_stop, raw_error_pct=100 * abs(delta_V) * r_stop)


def recommend_positions(x_grid, E, tol_pct=3.0, drift_um=2.0, min_E_frac=0.5, x_limits=None, n_best=3,
                        edge_exclude_um=0.0, valid=None, min_spacing_um=0.0):
    """Imaging positions: large E, and E flat to `tol_pct` over +/- drift_um of laser drift.

    `edge_exclude_um` keeps candidates away from the ends of the measured span, where the
    interpolation is least constrained (the leave-one-out test is worst at the end stops);
    `valid` is an optional boolean mask on x_grid of where the interpolation is trusted.
    Returns dict(x_grid, E, variation_pct, ok, recommended [best ok], best [top n]).
    """
    x_grid = np.asarray(x_grid, float); E = np.asarray(E, float)
    if edge_exclude_um > 0 and x_limits is None:
        x_limits = (x_grid.min() + edge_exclude_um, x_grid.max() - edge_exclude_um)
    elif edge_exclude_um > 0:
        x_limits = (x_limits[0] + edge_exclude_um, x_limits[1] - edge_exclude_um)
    var = np.full_like(E, np.nan)
    for i, x in enumerate(x_grid):
        m = np.abs(x_grid - x) <= drift_um + 1e-9
        if m.sum() >= 2 and np.isfinite(E[m]).all() and E[i] > 0:
            var[i] = 100 * (E[m].max() - E[m].min()) / E[i]
    allowed = np.isfinite(var)
    if x_limits is not None:
        allowed &= (x_grid >= x_limits[0]) & (x_grid <= x_limits[1])
    if valid is not None:
        allowed &= np.asarray(valid, bool)
    Emax = np.nanmax(np.where(allowed, E, np.nan)) if allowed.any() else np.nanmax(E)
    ok = allowed & (var <= tol_pct) & (E >= min_E_frac * Emax)
    order = np.argsort(-np.where(ok, E, -np.inf))
    best = []
    for i in order:
        if not ok[i] or len(best) >= n_best:
            break
        if all(abs(x_grid[i] - b["stage_um"]) >= min_spacing_um for b in best):
            best.append(dict(stage_um=float(x_grid[i]), E=float(E[i]), variation_pct=float(var[i])))
    # if nothing passes, offer the flattest allowed points above the E floor as fallbacks
    fb = allowed & (E >= min_E_frac * Emax)
    fo = np.argsort(np.where(fb, var, np.inf))
    fallback = [dict(stage_um=float(x_grid[i]), E=float(E[i]), variation_pct=float(var[i]))
                for i in fo[:n_best] if fb[i]]
    return dict(x_grid=x_grid, E=E, variation_pct=var, ok=ok, allowed=allowed,
                recommended=(best[0] if best else None), best=best, fallback=fallback)


# ----------------------------------------------------------------------------------- #
#  one call for the notebook                                                           #
# ----------------------------------------------------------------------------------- #
def analyse_survey(series, cfg, f_cr1_Hz=None, invols_by_x=None, x_limits=None, x_grid=None, verbose=True):
    """Everything the map route produces from one two-domain survey, as one dict.

    `f_cr1_Hz`: measured CR1 (pre-flight); None = found from the reference condition.
    `invols_by_x`: {stage x: InvOLS m/V} from the capture records; missing -> d_eff flagged.
    `x_grid`: fixed output grid (the live view uses the reachable span so successive E(x) compare).
    """
    p, a, c = cfg.probe, cfg.analysis, cfg.capture
    if f_cr1_Hz is None:
        f_cr1_Hz = find_cr1(series, p.cr_bands_Hz["cr1"])
    cr_band = (a.cr_band_frac[0] * f_cr1_Hz, a.cr_band_frac[1] * f_cr1_Hz)
    tab = channel_table(series, c.spots, p.qs_band_Hz, cr_band, track_peaks=a.track_peaks,
                        ref_bias_V=c.ref_bias_V, invols_by_x=invols_by_x)
    delta = dict(delta_V=0.0, dV_domains_V=0.0, D=np.nan, rms_resid_frac=np.nan, n_used=0)
    if a.fit_delta:
        mask = None
        if a.delta_exclude_tip_um > 0:
            mask = tab["x_um"] <= tab["x_um"].max() - a.delta_exclude_tip_um
        try:
            delta = fit_delta(tab, mask)
        except ValueError as e:
            if verbose: print("  delta fit skipped:", e)
    corr = apply_delta(tab, delta["delta_V"])
    emap = emap_along_lever(series, c.spots, p.qs_band_Hz, cr_band, delta_V=delta["delta_V"], rank=a.rank,
                            grid_step_um=a.grid_step_um, x_grid=x_grid, align=a.track_peaks, ref_bias_V=c.ref_bias_V)
    inv = tab["invols"]
    deff_qs = deff_qs_pm_per_V(np.abs(corr["P_qs_corr"]), inv, c.drive_V, p.amp_divisor)
    E_loo_at = np.interp(tab["x_um"], emap["x_meas"], emap["E_loo"]) if emap["x_meas"].size else np.nan
    deff_cr = deff_cr_pm_per_V(np.abs(corr["P_cr_corr"]), inv, c.drive_V, E_loo_at, p.amp_divisor)
    v_cpd = float(np.nanmedian(tab["v_cpd_qs"]))
    xm = emap["x_meas"]
    edge = cfg.imaging.edge_exclude_um
    if edge is None:                      # default: half the mean spacing of the measured positions
        edge = 0.5 * float(np.mean(np.diff(np.sort(xm)))) if xm.size > 1 else 0.0
    # only recommend where the interpolation is validated: both bracketing measured positions
    # must pass the leave-one-out test (|E_meas / E_loo - 1| <= loo_tol)
    valid = None
    if np.isfinite(emap["loo_ratio"]).any():
        order = np.argsort(xm); xs_, ok_ = xm[order], np.abs(emap["loo_ratio"][order] - 1) <= cfg.imaging.loo_tol
        valid = np.zeros(emap["x_grid"].size, bool)
        for i in range(xs_.size - 1):
            if ok_[i] and ok_[i + 1]:
                valid |= (emap["x_grid"] >= xs_[i]) & (emap["x_grid"] <= xs_[i + 1])
    rec = recommend_positions(emap["x_grid"], emap["E"], cfg.imaging.tol_pct, cfg.imaging.drift_um,
                              cfg.imaging.min_E_frac, x_limits, n_best=max(cfg.imaging.n_be_positions, 3),
                              edge_exclude_um=edge, valid=valid, min_spacing_um=cfg.imaging.be_min_spacing_um)
    esbs = qs_blind_spot(tab, delta["delta_V"], emap["x_grid"], rank=a.rank)
    # raw-vs-corrected QS d_eff interpolated to the blind spot: should agree there
    okc = tab["complete"] & np.isfinite(deff_qs)
    raw_q = deff_qs_pm_per_V(np.abs(tab["P_qs"]), inv, c.drive_V, p.amp_divisor)
    xs_ok = tab["x_um"][okc]; o = np.argsort(xs_ok)
    esbs["deff_raw_at_esbs"] = float(np.interp(esbs["x_esbs_um"], xs_ok[o], raw_q[okc][o])) if okc.sum() > 1 else np.nan
    esbs["deff_corr_at_esbs"] = float(np.interp(esbs["x_esbs_um"], xs_ok[o], deff_qs[okc][o])) if okc.sum() > 1 else np.nan
    esbs["E_at_esbs"] = float(np.interp(esbs["x_esbs_um"], emap["x_grid"], emap["E"]))
    out = dict(f_cr1_Hz=float(f_cr1_Hz), cr_band_Hz=cr_band, qs_band_Hz=tuple(p.qs_band_Hz), esbs=esbs,
               table=tab, delta=delta, corr=corr, emap=emap, deff_qs_pm_per_V=deff_qs, deff_cr_pm_per_V=deff_cr,
               v_cpd_V=v_cpd, v_cpd_spread_V=float(np.nanmax(tab["v_cpd_qs"]) - np.nanmin(tab["v_cpd_qs"])),
               recommend=rec, invols_known=bool(np.all(tab["invols_known"])))
    if verbose:
        print(summary_text(out))
    return out


def summary_text(res):
    t, d, e = res["table"], res["delta"], res["emap"]
    L = [f"CR1 {res['f_cr1_Hz'] / 1e3:.2f} kHz; analysis band {res['cr_band_Hz'][0] / 1e3:.0f}-"
         f"{res['cr_band_Hz'][1] / 1e3:.0f} kHz; QS band {res['qs_band_Hz'][0] / 1e3:.0f}-{res['qs_band_Hz'][1] / 1e3:.0f} kHz",
         f"Q (median over positions) {np.nanmedian(t['Q']):.0f};  V_cpd (QS, median) {res['v_cpd_V']:+.2f} V "
         f"(spread {res['v_cpd_spread_V']:.2f} V)",
         f"domain-potential term delta = {d['delta_V']:+.3f} V  (domain potentials differ by {d['dV_domains_V']:+.2f} V; "
         f"QS fit residual {100 * d['rms_resid_frac']:.1f} %)" if d["n_used"] else "no domain-potential correction",
         "",
         f"{'x (um)':>8} {'InvOLS':>10} {'f_CR1 kHz':>10} {'Q':>6} {'E_meas':>8} {'E_loo':>8} {'ratio':>6} "
         f"{'d_QS pm/V':>10} {'d_CR1/E':>8}"]
    Eloo = np.interp(t["x_um"], e["x_meas"], e["E_loo"]) if e["x_meas"].size else np.full(t["x_um"].size, np.nan)
    for j in range(t["x_um"].size):
        if not t["complete"][j]:
            L.append(f"{t['x_um'][j]:8.1f}   incomplete"); continue
        L.append(f"{t['x_um'][j]:8.1f} {t['invols'][j]:10.3e} {t['f_cr_Hz'][j] / 1e3:10.2f} {t['Q'][j]:6.0f} "
                 f"{res['corr']['E_meas'][j]:8.1f} {Eloo[j]:8.1f} {res['corr']['E_meas'][j] / Eloo[j]:6.2f} "
                 f"{res['deff_qs_pm_per_V'][j]:10.2f} {res['deff_cr_pm_per_V'][j]:8.2f}")
    ratios = res["corr"]["E_meas"] / Eloo
    L.append("")
    L.append(f"E_meas / E_loo: median {np.nanmedian(ratios):.2f} (range {np.nanmin(ratios):.2f}-{np.nanmax(ratios):.2f}) "
             f"-- the reconstruction interpolates E to this accuracy at never-visited positions")
    L.append(f"d_eff QS (corrected): median {np.nanmedian(res['deff_qs_pm_per_V']):.2f} pm/V, "
             f"sd {100 * np.nanstd(res['deff_qs_pm_per_V']) / np.nanmedian(res['deff_qs_pm_per_V']):.1f} %"
             + ("" if res["invols_known"] else "   ** InvOLS unknown at some positions: d_eff there is nominal **"))
    es = res["esbs"]
    L.append(f"QS stop (electrostatic blind spot): stage x = {es['x_esbs_um']:.1f} um "
             + (f"(zero crossing of b/P)" if es["crossing"] else
                f"(no zero crossing in the measured span -- the end with the smallest electrostatic fraction)")
             + f"; |b/P| there {es['r_at_stop']:.3f} -> an uncorrected QS image reads {es['raw_error_pct']:.1f} % off; "
             f"raw vs corrected QS d_eff at that stop {es['deff_raw_at_esbs']:.2f} vs {es['deff_corr_at_esbs']:.2f} pm/V")
    if res["recommend"]["best"]:
        L.append("Recommended BE stops (largest validated E, flat to spec, spaced): " + "; ".join(
            f"x = {b['stage_um']:.0f} um, E = {b['E']:.0f} (+/-{b['variation_pct']:.1f} %)" for b in res["recommend"]["best"]))
    else:
        L.append("No position satisfies the flatness + E criteria; the flattest allowed stops above the E floor are:")
        for f in res["recommend"]["fallback"]:
            L.append(f"   x = {f['stage_um']:.0f} um, E = {f['E']:.0f}, varies {f['variation_pct']:.1f} %  "
                     f"(pass x_override= to pick_position, or raise imaging.tol_pct)")
    return "\n".join(L)
