"""Complex-field reconstructors: amplitude AND phase, with a predictive sigma.

The published arms (physrec.py) work on |Z| only -- the FEM/EB library was
reduced to log-amplitude at build time and the measured phase was discarded in
`run_phys.prep`.  This module keeps the complex field end to end:

  * the physics parameters (k1, f_res, g) are fitted with the EXACT published
    objective, `physrec.fit_eb` on log(A+F) over the revealed SNR window, so
    the fitted contact stiffness here is bit-identical to the deck's;
  * the FORWARD PREDICTION is then taken from the complex channel of the
    library (`re_u`, `im_u`) instead of its log-amplitude channel, and scaled
    by one complex gain fitted to the revealed block.  The gain is what absorbs
    the instrument's arbitrary reference phase and the pm-per-volt scale;
  * the discrepancy GP is complex: the same isotropic squared-exponential over
    position, applied to the real and imaginary residual with a shared length
    scale chosen by leave-one-out PRESS on the revealed points.

Leakage rule is unchanged: reconstructors see only (x_grid, sel_idx, Z_sel,
freq).  Everything -- noise floor, SNR window, complex gain, GP length scale,
per-frequency variance -- is estimated from the revealed block.

Two uncertainty channels are returned and they mean different things:
  sd_par   parameter uncertainty, from a leave-one-out refit of (k1, f_res, g);
           it shrinks as positions are added and is largest where the model is
           most sensitive to the contact stiffness (the node, the antiresonance)
  sd_gp    discrepancy-GP posterior sd; it is the design-driven term, largest
           far from any revealed position, and it does NOT shrink where the
           model happens to be right
`sd` is the quadrature sum, per real component.
"""
import numpy as np
import physrec as PR

# ---------------------------------------------------------------- library
_LIB = {}


def load(path):
    """Load the complex channel of an aligned library and point physrec at it.

    physrec.use_library rebinds the log-amplitude globals used by the fit; this
    adds the Re/Im blocks on the same (k1, damping, u, position) grid so the
    forward prediction can be complex.  Both must come from the SAME file or
    the fit and the prediction would describe different forward models.
    """
    msg = PR.use_library(path)
    L = np.load(path)
    if 're_u' not in L.files:
        raise ValueError(f'{path} has no complex channel -- rebuild it with '
                         'build_femlib_cplx.py')
    _LIB['re'] = np.ascontiguousarray(L['re_u'])
    _LIB['im'] = np.ascontiguousarray(L['im_u'])
    _LIB['UG'] = L['UG']
    return msg


def predict_cplx(log_k1, f_res, log_g, freq, rows=None):
    """Complex model field (npos, nfreq), same interpolation as eb_predict.

    Cubic in log k1, linear in log g and in the resonance-aligned frequency
    u = f / f_res -- identical weights to the log-amplitude path, applied to
    Re and Im.  Linear interpolation of a linear quantity, so this is the
    honest complex analogue; note |predict_cplx| is NOT exp(eb_predict), and
    cannot be, because |.| does not commute with interpolation across a null.
    """
    ik, wk = PR._ax(PR.LK1, log_k1)
    ig, wg = PR._ax(PR.LGG, log_g)
    UG = _LIB['UG']
    u = np.clip(np.asarray(freq, float) / f_res, UG[0], UG[-1])
    ju = np.clip(np.searchsorted(UG, u) - 1, 0, len(UG) - 2)
    wu = ((u - UG[ju]) / (UG[ju + 1] - UG[ju]))[:, None]
    kw = PR._cubic_w(wk)
    kidx = [np.clip(ik - 1 + d, 0, PR.NK - 1) for d in range(4)]
    out = 0.0
    for a, wa in zip(kidx, kw):
        if wa == 0.0:
            continue
        for b, wb in ((ig, 1 - wg), (ig + 1, wg)):
            if wb == 0.0:
                continue
            for blk, mul in ((_LIB['re'][a, b], 1.0), (_LIB['im'][a, b], 1j)):
                B = blk if rows is None else blk[:, rows]
                out = out + mul * wa * wb * ((1 - wu) * B[ju] + wu * B[ju + 1])
    return np.asarray(out).T


# ---------------------------------------------------------------- complex GP
def _press_cplx(xs, R, ell, noise):
    """LOO PRESS with the complex residual stacked as [Re | Im]."""
    R2 = np.concatenate([R.real, R.imag], axis=1)
    return PR._press(xs, R2, ell, noise)


def _gp_posterior(x_grid, xs, R, ell, noise_frac=0.05):
    """Per-frequency SE-GP posterior mean and sd for a complex residual.

    The length scale is shared across frequency (one number, chosen by PRESS);
    the signal variance is NOT -- v_f is the residual variance at that
    frequency, so the returned sd tracks where the discrepancy actually lives
    (large through the resonance and across the node, small off resonance)
    instead of smearing one global number over the whole band.
    """
    n = len(xs)
    d2 = (xs[:, None] - xs[None, :]) ** 2
    Kc = np.exp(-d2 / (2 * ell ** 2))
    kc = np.exp(-((x_grid[:, None] - xs[None, :]) ** 2) / (2 * ell ** 2))
    A = Kc + noise_frac * np.eye(n)
    Ai = np.linalg.inv(A)
    # leverage-driven shape, identical for every frequency
    shape = np.clip(1.0 - np.einsum('ij,jk,ik->i', kc, Ai, kc), 0.0, None)
    v = R.real.var(axis=0) + R.imag.var(axis=0) + 1e-30      # per-frequency
    v = 0.5 * v                                              # per component
    mean = kc @ (Ai @ R)                                     # (npos, nfreq)
    sd = np.sqrt(np.outer(shape, v))                         # (npos, nfreq)
    return mean, sd


# ---------------------------------------------------------------- arms
def _gain(Ms, Zs, F):
    """One complex gain: detection scale x arbitrary reference phase.

    NOT the complex least-squares gain <M,Z>/<M,M>.  The measured phase runs
    ~180 deg across the resonance, and any residual phase error between model
    and data makes that coherent sum cancel -- the LS gain came out ~10x too
    small and put the arm at 100 % error, with the amplitude wrong for a reason
    that had nothing to do with amplitude.  Instead the two halves are
    estimated separately, each with the estimator that is robust for it:

      |gain|   the published log-amplitude gain, a mean of log ratios over the
               high-SNR revealed bins -- the same number rec_eb uses, so the
               amplitude of this arm matches the published FEM arm
      arg gain the amplitude-weighted CIRCULAR mean of arg(Z/M), which is the
               right average for an angle and is not destroyed by the 180 deg
               swing through resonance
    """
    w = np.abs(Zs) > 3.0 * F
    if w.sum() < 8:
        w = np.ones_like(np.abs(Zs), bool)
    lr = np.log(np.abs(Zs)[w]) - np.log(np.maximum(np.abs(Ms)[w], 1e-300))
    mag = float(np.exp(np.mean(lr)))
    r = (Zs / Ms)[w]
    ang = float(np.angle(np.sum(np.abs(Zs)[w] * r / np.abs(r))))
    return mag * np.exp(1j * ang)


def rec_gp_cplx(x_grid, sel_idx, Z_sel, freq,
                ells=(1.5, 2.5, 4., 8., 15., 30., 60., 120.), **kw):
    """GP alone on the complex field: per-frequency complex constant mean."""
    sel = np.asarray(sel_idx)
    n = len(sel)
    m = Z_sel.mean(axis=0)
    R = Z_sel - m[None, :]
    xs = x_grid[sel]
    noise = 0.05
    ell = min(ells, key=lambda e: _press_cplx(xs, R, e, noise * float(
        np.mean(np.abs(R) ** 2)) + 1e-18))
    corr, sd = _gp_posterior(x_grid, xs, R, ell)
    Zrec = np.repeat(m[None, :], len(x_grid), axis=0) + corr
    return dict(Zrec=Zrec, sd=sd, sd_gp=sd, sd_par=np.zeros_like(sd),
                ell=ell, theta={}, gain=None)


def fit_cplx(sel_idx, Z_sel, freq, F, win, th0):
    """Refine (f_res, g) against the COMPLEX residual on the revealed window.

    Leakage-safe: nothing outside `sel_idx` is touched.  The amplitude fit pins
    the resonance only to the accuracy with which a peak can be located, ~0.1 %
    here; on a Q ~ 200 line 0.1 % of f_res is a fifth of the half-width, which
    is tens of degrees of phase.  Phase is the sensitive channel for (f_res, Q)
    and amplitude is the sensitive channel for the mode shape, so this refines
    exactly the two parameters the amplitude fit constrains worst and leaves
    k1 -- which the shape pins -- where the published fit put it.
    """
    from scipy.optimize import minimize_scalar
    sel = np.asarray(sel_idx)
    Zs = Z_sel[:, win]
    fw = freq[win]

    def cost(fres, lg):
        M = predict_cplx(th0['log_k1'], fres, lg, fw, rows=sel)
        g = _gain(M, Zs, F)
        return float(np.sum(np.abs(g * M - Zs) ** 2) / np.sum(np.abs(Zs) ** 2))

    fres, lg = th0['f_res'], th0['log_g']
    best = (cost(fres, lg), fres, lg)
    for fr in fres * np.linspace(0.995, 1.005, 41):
        for g_ in np.linspace(PR.LGG[0], PR.LGG[-1], 9):
            c = cost(fr, g_)
            if c < best[0]:
                best = (c, fr, g_)
    _, fres, lg = best
    for _ in range(3):
        r = minimize_scalar(lambda v: cost(v, lg),
                            bounds=(fres * 0.999, fres * 1.001),
                            method='bounded', options={'xatol': 1e-3})
        fres = float(r.x)
        r = minimize_scalar(lambda v: cost(fres, v),
                            bounds=(PR.LGG[0], PR.LGG[-1]),
                            method='bounded', options={'xatol': 1e-5})
        lg = float(r.x)
    th = dict(th0)
    th.update(f_res=fres, log_g=lg, g=float(np.exp(lg)),
              f_free=PR.f_free_of(th0['log_k1'], fres),
              cplx_rms=float(np.sqrt(cost(fres, lg))))
    return th


def rec_phys_cplx(x_grid, sel_idx, Z_sel, freq, gp=True, cplx_fit=False,
                  ells=(4., 8., 15., 30., 60., 120.), loo=True,
                  fit_band=None, **kw):
    """Physics forward model (+ complex discrepancy GP) on the complex field.

    `fit_band = (lo, hi)` restricts the FIT window to that frequency range.
    Necessary whenever the supplied frequency axis runs outside the library's
    u coverage: physrec.snr_window keys off 10 % of the revealed maximum, and on
    the wide band mode 2 clears that threshold at 57 % of mode 1 -- so a band-A
    library gets asked to fit a resonance it cannot represent (predict_cplx
    clips at u = 2.05) and the fitted k1 goes anywhere from 111 to 2710 N/m.
    Restricting the fit to the band the library covers puts k1 back on the
    published value; the SCORE can then still be taken over the whole coverage.
    """
    sel = np.asarray(sel_idx)
    A_sel = np.abs(Z_sel)
    F = PR.noise_floor(A_sel)
    win = PR.snr_window(A_sel, freq)
    if fit_band is not None:
        win = win & (freq >= fit_band[0]) & (freq <= fit_band[1])
    th = PR.fit_eb(sel, A_sel, freq, F=F, win=win)           # published fit
    if cplx_fit:
        th = fit_cplx(sel, Z_sel, freq, F, win, th)

    def field(t, rows=sel, Zs_full=Z_sel):
        M = predict_cplx(t['log_k1'], t['f_res'], t['log_g'], freq)
        Ms, Zs = M[rows][:, win], Zs_full[:, win]
        gam = _gain(Ms, Zs, F)
        return gam * M, gam

    Zm, gam = field(th)

    # --- parameter uncertainty: leave-one-out refit, exactly as rec_eb does
    sd_par = np.zeros(Zm.shape)
    if loo and len(sel) >= 4:
        folds = (list(range(len(sel))) if len(sel) <= 12
                 else list(np.linspace(0, len(sel) - 1, 12).astype(int)))
        P = []
        for j in folds:
            k = [i for i in range(len(sel)) if i != j]
            t2 = PR.fit_eb([sel[i] for i in k], A_sel[k], freq, F=F, win=win,
                           coarse=(8, 7, 3), refine=1)
            M2 = predict_cplx(t2['log_k1'], t2['f_res'], t2['log_g'], freq)
            g2 = _gain(M2[sel][:, win][k], Z_sel[k][:, win], F)
            P.append(g2 * M2)
        P = np.asarray(P)
        sd_par = np.sqrt(0.5 * (P.real.var(0) + P.imag.var(0)))

    if not gp:
        return dict(Zrec=Zm, sd=np.maximum(sd_par, 1e-30), sd_gp=np.zeros_like(sd_par),
                    sd_par=sd_par, ell=None, theta=th, gain=gam)

    R = Z_sel - Zm[sel]
    xs = x_grid[sel]
    if len(sel) < 5:
        return dict(Zrec=Zm, sd=np.maximum(sd_par, 1e-30),
                    sd_gp=np.zeros_like(sd_par), sd_par=sd_par, ell=None,
                    theta=th, gain=gam)
    ell = min(ells, key=lambda e: _press_cplx(
        xs, R, e, 0.05 * float(np.mean(np.abs(R) ** 2)) + 1e-18))
    corr, sd_gp = _gp_posterior(x_grid, xs, R, ell)
    return dict(Zrec=Zm + corr, sd=np.sqrt(sd_par ** 2 + sd_gp ** 2),
                sd_gp=sd_gp, sd_par=sd_par, ell=ell, theta=th, gain=gam)


# ---------------------------------------------------------------- scoring
def score_cplx(x_grid, sel_idx, out, Z_true, freq, truth_dns=np.nan):
    from activemodemap.lowrank import classify_null_from_map, resonance_index
    held = np.array([i for i in range(len(x_grid)) if i not in set(sel_idx)])
    Zr, Zt = out['Zrec'][held], Z_true[held]
    m = {}
    m['nrmse_cplx'] = 100 * np.linalg.norm(Zr - Zt) / np.linalg.norm(Zt)
    m['nrmse_amp'] = (100 * np.linalg.norm(np.abs(Zr) - np.abs(Zt))
                      / np.linalg.norm(np.abs(Zt)))
    # phase error, amplitude-weighted (phase is meaningless where |Z| ~ noise)
    w = np.abs(Zt)
    dph = np.angle(Zr * np.conj(Zt))
    m['phase_mae_deg'] = float(np.degrees(np.sum(w * np.abs(dph)) / w.sum()))
    ir = resonance_index(freq, Z_true[sel_idx])
    cl = classify_null_from_map(x_grid, out['Zrec'], freq, ir)
    d = cl['x_null_um'] if np.isfinite(cl['x_null_um']) else cl['x_bound_um']
    m['dns'] = d
    m['dns_err'] = abs(d - truth_dns) if np.isfinite(d) else np.nan
    # calibration on held-out points, per real component
    S = out['sd'][held]
    ok = S > 0
    z = np.concatenate([((Zr - Zt).real / S)[ok], ((Zr - Zt).imag / S)[ok]])
    m['z_rms'] = float(np.sqrt(np.mean(z ** 2)))
    for k, nom in ((1, .6827), (2, .9545), (3, .9973)):
        m[f'cov{k}s'] = float(np.mean(np.abs(z) <= k))
        m[f'cov{k}s_nom'] = nom
    return m, z


# ------------------------------------------------- two-band (modes 1 and 2)
def rec_twoband_cplx(x_grid, sel_idx, Z_sel, freq, band_b_lo=700e3, gp=True,
                     ells=(4., 8., 15., 30., 60., 120.), loo=False,
                     fit_band=(265e3, 1000e3), coarse=(48, 15, 7), **kw):
    """Physics arm fitted across BOTH flexural bands, on the complex field.

    Parameters come from `twoband.fit2g` -- the published leakage-safe two-band
    fit with a PER-BAND gain, which is what pins k1 to ~1000 N/m instead of the
    ~390 N/m that band A alone prefers.  The per-band gain is not a fudge: the
    electrostatic drive is distributed and couples 3.11x less to mode 2, so the
    RELATIVE modal amplitude genuinely needs one scalar per band.

    The complex prediction then takes a per-band COMPLEX gain, estimated with
    the same robust magnitude/circular-phase split used by rec_phys_cplx.  A
    single gain across both bands would be wrong for the same reason it is
    wrong in log-amplitude, and additionally in phase: the two bands do not
    share a detection phase once the drive coupling differs.

    Requires a library whose u range reaches mode 2 (eblib_x_cplx.npz).
    """
    import twoband as TB
    sel = np.asarray(sel_idx)
    A_sel = np.abs(Z_sel)
    # Fit on the two-band window the published spec uses, 265-1000 kHz.  Handing
    # fit2g the whole 25-1775 kHz axis instead lets its band-B SNR window reach
    # the 970 kHz feature and 800 kHz of noise, and k1 comes out 494-533 N/m at
    # small n instead of locking to ~1008.
    fb = (freq >= fit_band[0]) & (freq <= fit_band[1])
    saved = TB.BAND_B
    TB.BAND_B = (band_b_lo, float(fit_band[1]))
    try:
        # fit2g's default 16-point coarse grid in log k1 is too sparse for this
        # cost surface: at n = 5-8 it settles at k1 ~ 500 N/m while k1 = 1008
        # scores 0.35 against 0.61 -- a worse basin the coordinate refinement
        # cannot leave. 48 points finds the right one at every n. (The published
        # two-band run got ~1008 at every n with 16 points; that was the grid
        # happening to straddle the good basin, not robustness.)
        th = TB.fit2g(sel, A_sel[:, fb], freq[fb], coarse=coarse)
    finally:
        TB.BAND_B = saved
    # fit2g returns windows on the restricted axis; lift them back to the full one
    for k in ('win', 'wa', 'wb'):
        full = np.zeros(len(freq), bool)
        full[np.where(fb)[0]] = th[k]
        th[k] = full
    F = th['F']
    inb = freq >= band_b_lo
    M = predict_cplx(th['log_k1'], th['f_res'], th['log_g'], freq)
    gam = np.ones(len(freq), complex)
    for m, wm in ((~inb, th['wa']), (inb, th['wb'])):
        if wm.sum() < 4:
            m2 = m
        else:
            m2 = wm
        gam[m] = _gain(M[sel][:, m2], Z_sel[:, m2], F)
    Zm = gam[None, :] * M
    sd_par = np.zeros(Zm.shape)
    if not gp or len(sel) < 5:
        return dict(Zrec=Zm, sd=np.full(Zm.shape, 1e-30), sd_gp=sd_par,
                    sd_par=sd_par, ell=None, theta=th, gain=gam)
    R = Z_sel - Zm[sel]
    xs = x_grid[sel]
    ell = min(ells, key=lambda e: _press_cplx(
        xs, R, e, 0.05 * float(np.mean(np.abs(R) ** 2)) + 1e-18))
    corr, sd_gp = _gp_posterior(x_grid, xs, R, ell)
    return dict(Zrec=Zm + corr, sd=sd_gp, sd_gp=sd_gp, sd_par=sd_par,
                ell=ell, theta=th, gain=gam)


def coverage(freq, f_res):
    """Which frequencies the loaded library can actually predict.

    predict_cplx CLIPS u to the library's range, so outside coverage it returns
    the edge of the library rather than nothing -- silently plausible and
    completely wrong.  Every wide-band score must mask on this.
    """
    u = np.asarray(freq, float) / f_res
    UG = _LIB['UG']
    return (u >= UG[0]) & (u <= UG[-1])
