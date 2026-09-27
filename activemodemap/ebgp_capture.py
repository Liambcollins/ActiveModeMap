"""EB+GP (physics-informed) active-learning capture driver.

The model-free counterpart is `activemodemap.sparse_capture.run_sparse_capture`,
which drives `LowRankModeMap` and picks positions D-optimally on a Chebyshev
basis. That selector clustered 30 picks into ~7 groups on this probe. This driver
does the same job with the corrected physics path instead:

    PhysicsPosterior(fit_zeta=True, fit_geometry="setback", analytic_gain=True)

fitted to the CR1 band after every point, with the next position chosen by the
D-optimal expected-information-gain criterion on theta (`acquisition.acquire_eig`,
reimplemented here for a SINGLE measured domain and restricted to the reachable
1 um candidate grid).

Two differences from `acquisition.acquire_eig` that matter:
  * `_jacobian_at_x` stacks both domains (+1 and -1). This experiment measures one
    domain, so the (-1) rows are information we will never collect and they carry
    their own x-dependence. `_jacobian_single` below uses the measured domain only.
  * `_candidates` walks its own xi grid from 0.15 and would propose positions off
    the reachable span. Candidates here are the measurement grid itself.

The posterior object is reused across the loop, so `fit()` warm-starts from the
previous MAP (see `PhysicsPosterior.fit`) instead of doing a cold multi-start
every step.

`measure_fn(x_um) -> (freq_Hz, Z)` is the only instrument coupling, so the same
loop replays offline against a dense map.
"""

from __future__ import annotations

import time
import numpy as np

from activemodemap.forward_model import EBForwardModel, ProbeGeometry
from activemodemap.inference import PhysicsPosterior

OM1 = 1.87510407 ** 2

# PPP-CONTAu geometry, as established in scripts/eb_gp_dense/bandcurve.py
F0_HZ, K_LEVER, L_UM, TIP_MASS_RATIO = 13.649e3, 0.4057, 445.0, 0.0018
SETBACK_BOUNDS_UM = (4.0, 20.0)

# Seed design. `loop.INIT_POSITIONS` is (0.35, 0.70, 0.95) in xi, which on the
# reachable span (x = 100-445 um, xi = 0.225-1.0) puts all three seeds in the outer
# half. Fitted from those three, the geometry is not identifiable: the setback rails
# at its 20 um upper bound and zeta collapses to ~0.0004 (verified on the Sep-19
# dense map). Seeding at the span ENDS instead -- which is what subset_indices()
# does for every validated fit in scripts/eb_gp_dense/bandcurve.py -- recovers
# setback 9.75 um / zeta 0.0047 on the same data. Seeds are therefore taken as
# evenly spaced fractions OF THE REACHABLE GRID, not of the lever.
SEED_FRAC = (0.0, 0.5, 1.0)


def build_model(f_band_Hz, geom=None):
    """EB model on the measured band's own frequency grid."""
    m = EBForwardModel(
        geom=geom or ProbeGeometry(name='PPP-CONTAu', f0_hz=F0_HZ, k_lever=K_LEVER,
                                   L_um=L_UM, tip_setback_um=10.0, tip_height_um=12.5,
                                   tilt_deg=11.0, tip_mass_ratio=TIP_MASS_RATIO),
        n_modes=12, nx=241, nf=f_band_Hz.size,
        omega_lo=float(f_band_Hz[0] * OM1 / F0_HZ),
        omega_hi=float(f_band_Hz[-1] * OM1 / F0_HZ))
    m.PRIOR_LO = np.array([1.0, 1.0, 0.5, -4.0, -4.0])
    m.PRIOR_HI = np.array([5.0, 4.0, 2.5, 1.0, 1.0])
    return m


def to_model_grid(f_meas_Hz, Z_rows, model):
    """Interpolate measured in-band spectra onto the model's frequency grid."""
    f_model = model.omega / OM1 * model.geom.f0_hz
    out = np.empty((len(Z_rows), f_model.size), complex)
    for i, z in enumerate(Z_rows):
        out[i] = (np.interp(f_model, f_meas_Hz, z.real, left=np.nan, right=np.nan)
                  + 1j * np.interp(f_model, f_meas_Hz, z.imag, left=np.nan, right=np.nan))
    if not np.isfinite(out).all():
        raise ValueError('model frequency grid runs outside the measured band')
    return out


def sigma_est(Z, half=6):
    """Noise scale: residual after a running mean, robustified with the MAD.

    Same estimator as scripts/eb_gp_dense/wideband.py.
    """
    from scipy.ndimage import uniform_filter1d
    r = Z - (uniform_filter1d(Z.real, 2 * half + 1, axis=1)
             + 1j * uniform_filter1d(Z.imag, 2 * half + 1, axis=1))
    a = np.abs(r).ravel()
    return float(1.4826 * np.median(np.abs(a - np.median(a))) + np.median(a))


# --------------------------------------------------------------- acquisition
def _jacobian_single(post, xi_val, dtheta=1e-3):
    """Whitened numeric Jacobian of the SINGLE-domain spectrum at xi wrt theta.

    Mirrors acquisition._jacobian_at_x but stacks only the measured (+1) domain,
    and uses the analytic complex gain when the posterior carries one.
    """
    m = post.model
    i = int(np.argmin(np.abs(m.xi - xi_val)))
    th0 = post.theta_map

    def stacked(th):
        resp = m.response(post._core(th))
        A0 = post.gain if post.analytic_gain else resp["A0"]
        z = A0 * (post._blur(resp["piezo"])[:, i]
                  + resp["eps"] * post._blur(resp["elec"])[:, i])
        return np.concatenate([z.real, z.imag])

    f0 = stacked(th0)
    nf = f0.size // 2
    amp = np.sqrt(f0[:nf] ** 2 + f0[nf:] ** 2)
    se = np.tile(post._sigma_eff(amp), 2)
    J = np.empty((f0.size, th0.size))
    for k in range(th0.size):
        th = th0.copy()
        th[k] += dtheta
        J[:, k] = (stacked(th) - f0) / dtheta
    return J / se[:, None]


def acquire_eig_on_grid(post, x_grid_um, measured_x_um, L_um=L_UM,
                        min_sep_um=4.0, n_cand=48, blocked=()):
    """D-optimal EIG, 0.5*logdet(I + Sigma J^T J), over the reachable grid.

    Candidates are drawn from `x_grid_um` itself (thinned to ~n_cand for cost),
    excluding anything within `min_sep_um` of an already-measured position. That
    exclusion is the same guard acquisition._candidates applies in xi (min_sep
    0.01 -> 4.45 um on this lever); it is stated in um here so it can be reported.
    """
    cand = np.asarray(x_grid_um, float)
    if len(measured_x_um):
        mx = np.asarray(measured_x_um, float)
        cand = cand[np.min(np.abs(cand[:, None] - mx[None, :]), axis=1) >= min_sep_um]
    for b in blocked:
        cand = cand[np.abs(cand - b) > 1e-9]
    if cand.size == 0:
        return None
    if cand.size > n_cand:                       # thin evenly, keep the ends
        cand = cand[np.unique(np.linspace(0, cand.size - 1, n_cand).astype(int))]

    p = len(post.theta_map)
    best_x, best = None, -np.inf
    for c in cand:
        Jw = _jacobian_single(post, c / L_um)
        sign, logdet = np.linalg.slogdet(np.eye(p) + post.cov @ (Jw.T @ Jw))
        if sign > 0 and logdet > best:
            best, best_x = logdet, float(c)
    return best_x


# ------------------------------------------------------------------ the loop
def run_ebgp_capture(measure_fn, x_grid_um, band_Hz=(50.2e3, 79.8e3), *,
                     max_pos=30, seed_frac=SEED_FRAC, min_sep_um=4.0, n_cand=48,
                     on_point=None, verbose=True):
    """Physics-driven capture. `measure_fn(x_um) -> (freq_Hz, Z)` or None on failure.

    `on_point(step, x_um, freq, Z)` is called after every successful measurement,
    before the fit -- use it to checkpoint the raw wideband data.

    Returns dict(x_sel, Z_band, post, model, history, blocked).
    """
    x_grid_um = np.asarray(x_grid_um, float)
    lo, hi = band_Hz
    xs, Zb, history, blocked = [], [], [], []
    post = model = f_band = None
    seed_i, consecutive_failures = 0, 0
    t0 = time.time()
    if verbose:
        print(f"EB+GP capture: band {lo/1e3:.1f}-{hi/1e3:.1f} kHz, max {max_pos} positions, "
              f"grid {x_grid_um[0]:.0f}-{x_grid_um[-1]:.0f} um ({x_grid_um.size} candidates), "
              f"min separation {min_sep_um:.0f} um", flush=True)

    def _nearest_unblocked(x_want):
        """Grid point nearest x_want that has not failed and is not already measured."""
        cand = x_grid_um
        taken = np.array(list(blocked) + list(xs), float)
        if taken.size:
            cand = cand[np.min(np.abs(cand[:, None] - taken[None, :]), axis=1) > 1e-9]
        return None if cand.size == 0 else float(cand[np.argmin(np.abs(cand - x_want))])

    def _maximin(x_measured):
        """Fallback selector when there is no usable posterior: the reachable
        grid point farthest from everything measured so far."""
        cand = x_grid_um
        for b in blocked:
            cand = cand[np.abs(cand - b) > 1e-9]
        if cand.size == 0:
            return None
        if not len(x_measured):
            return float(cand[0])
        d = np.min(np.abs(cand[:, None] - np.asarray(x_measured, float)[None, :]), axis=1)
        return float(cand[np.argmax(d)])

    while len(xs) < max_pos:
        # ---- choose the next position
        step = len(xs)
        if seed_i < len(seed_frac):
            # seeds are consumed by ATTEMPT, not by success: a failed seed must not
            # be retried forever at the same x
            j = int(round(seed_frac[seed_i] * (x_grid_um.size - 1)))
            seed_i += 1
            x_next = _nearest_unblocked(float(x_grid_um[j]))
            how = "seed"
        elif post is not None and post.theta_map is not None and post.cov is not None:
            x_next = acquire_eig_on_grid(post, x_grid_um, xs, min_sep_um=min_sep_um,
                                         n_cand=n_cand, blocked=blocked)
            how = "eig"
        else:
            x_next = _maximin(xs)
            how = "maximin (no posterior yet)"
        if x_next is None:
            print("no candidate left that satisfies the separation guard; stopping")
            break

        if verbose:
            print(f"\n=== position {step + 1}/{max_pos}: x = {x_next:.1f} um ({how}) ===",
                  flush=True)
        try:
            got = measure_fn(x_next)
        except Exception as e:
            print(f"  measure_fn raised {type(e).__name__}: {e}")
            got = None
        if got is None:
            blocked.append(x_next)
            consecutive_failures += 1
            print(f"  failed at x={x_next:.1f}; blocked ({consecutive_failures} consecutive)")
            if consecutive_failures >= 3:
                print("  THREE IN A ROW - stopping")
                break
            continue
        consecutive_failures = 0
        freq, Z = got
        if on_point is not None:
            try:
                on_point(step, x_next, freq, Z)
            except Exception as e:
                # a checkpoint hiccup must not kill a campaign that is mid-flight
                print(f"  on_point/checkpoint failed ({type(e).__name__}: {e}); continuing")

        # ---- crop to the band (the model grid is fixed on the first point)
        w = (freq >= lo) & (freq <= hi)
        if model is None:
            f_band = freq[w]
            model = build_model(f_band)
            if verbose:
                print(f"  model grid: {f_band.size} bins, {f_band[1]-f_band[0]:.0f} Hz")
        xs.append(x_next)
        Zb.append(np.interp(f_band, freq[w], Z[w].real)
                  + 1j * np.interp(f_band, freq[w], Z[w].imag))

        if len(xs) < 3:               # theta is not identifiable from 1-2 positions
            if verbose:
                print(f"  {len(xs)}/3 seed positions ({(time.time()-t0)/60:.1f} min)")
            continue

        # ---- fit the posterior on everything measured so far
        Zg = to_model_grid(f_band, Zb, model)
        norm = np.abs(Zg).max()
        Zn = Zg / norm
        sigma = sigma_est(Zn)
        if post is None:
            post = PhysicsPosterior(model, sigma=sigma, rng=np.random.default_rng(0),
                                    fit_zeta=True, fit_geometry='setback',
                                    analytic_gain=True,
                                    setback_bounds_um=SETBACK_BOUNDS_UM)
        post.sigma = sigma
        data = [{'x': xv / L_UM, 'plus': zr} for xv, zr in zip(xs, Zn)]
        tf = time.time()
        try:
            th = post.fit(data)
        except Exception as e:
            # keep the last good posterior for selection; the data is already
            # checkpointed, and the offline analysis does not depend on this fit
            print(f"  posterior fit failed at n={len(xs)} ({type(e).__name__}: {e}); "
                  f"keeping the previous MAP for selection", flush=True)
            history.append(dict(n=len(xs), x_um=x_next, how=how, fit_failed=str(e),
                                elapsed_min=(time.time() - t0) / 60))
            continue
        D = dict(zip(post.names, th))
        # in-sample complex residual of the EB mean against what we measured
        pred = post.predict_map()
        cols = [int(np.argmin(np.abs(post.model.xi - xv / L_UM))) for xv in xs]
        eb = pred[:, cols].T
        crmse = float(np.sqrt(np.mean(np.abs(eb - Zn) ** 2)) / np.abs(Zn).max())
        rec = dict(n=len(xs), x_um=x_next, how=how, crmse_eb_insample=crmse,
                   setback_um=float(D.get('tip_setback_um', np.nan)),
                   zeta=float(10 ** D['log_zeta']), gain_abs=float(abs(post.gain)),
                   red_chi2=float(post.red_chi2), fit_s=time.time() - tf,
                   elapsed_min=(time.time() - t0) / 60)
        history.append(rec)
        if verbose:
            print(f"  n={rec['n']:2d}  EB in-sample {crmse:.4f}  setback {rec['setback_um']:.2f} um  "
                  f"zeta {rec['zeta']:.4f}  red_chi2 {rec['red_chi2']:.2f}  "
                  f"(fit {rec['fit_s']:.0f}s, {rec['elapsed_min']:.1f} min)", flush=True)

    return dict(x_sel=xs, Z_band=np.array(Zb), f_band=f_band, post=post,
                model=model, history=history, blocked=blocked)
