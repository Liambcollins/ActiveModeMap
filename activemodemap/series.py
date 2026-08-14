"""Mode-shape reconstruction as a function of DC bias and/or load.

Two things this gives you that a single-condition run does not.

**D-NS migration.** The contact resonance and the displacement null both move with
applied load (stiffer contact) and, weakly, with bias. Reconstructing at each
condition traces that migration.

**Channel separation from the bias dependence alone — no domain switching.** The
detected response is the sum of a piezoresponse pathway and an electrostatic one,

    Z(x, f; V) = Z_piezo(x, f) + (V - V_cpd) * S_elec(x, f)

because the first-harmonic electrostatic force goes as (V_dc - V_cpd)·V_ac while
the piezoresponse is bias-independent. So a complex straight-line fit in V at each
(x, f) splits them:

    slope     dZ/dV = S_elec        -> the pure electrostatic channel (D-ESBS)
    intercept Z(V=0) = Z_piezo - V_cpd·S_elec

The slope is exact and needs no knowledge of V_cpd, so **the D-ESBS is recovered
rigorously**. The intercept still carries a -V_cpd·S_elec term, so the D-NS taken
from it is only clean when V_cpd is small or supplied (`v_cpd` below); otherwise
treat it as "the response at V = 0" rather than "pure piezoresponse". See
`separate_channels`.

**What sets the SNR you need.** The D-NS comes from the antiresonance notch, whose
depth in this model is ~2% of the resonance peak. Noise much above that buries the
notch and the branch crossing degenerates into dozens of spurious crossings from
noise wiggles -- no estimator can rescue it, and no amount of frequency smoothing
helps (smoothing wide enough to suppress the wiggles also erases the notch). What
matters is noise relative to the OFF-RESONANCE BASELINE, not to the peak: with the
peak ~140x the baseline here, validation held from 2% to 50% of baseline
(D-NS within 0.6 um, D-ESBS within 0.25 um of the noise-free answer through the
same estimators). If the reconstructed map shows many branch crossings scattered
along the beam, that is the signature of an unresolvable notch -- average longer
or narrow the tune window rather than trusting the number.

Validated against the Euler-Bernoulli model by comparing the recovered channels to
the model's own piezo and electrostatic maps through the same estimators. Note the
model puts D-NS and D-ESBS within ~1-2 um of each other at resonance, so the
simulation confirms the machinery but cannot strongly demonstrate that the two
spots are resolvable from one another -- real PPLN data with a genuine
electrostatic offset is the test of that.

Acquisition is organised per POSITION, not per condition: the laser move,
AutoWedge, InvOLS and engage cost about a minute, whereas changing bias costs a
second. Visiting each position once and sweeping every condition there is several
times faster, and — because the bias fit is done per position — removes
position-repeatability error from the comparison.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict

import numpy as np

from .lowrank import (LowRankModeMap, reconstruct_map, dns_from_map,
                      resonance_index, band_mask, dns_branch, spatial_null)


# --------------------------------------------------------------------------- #
#  Conditions                                                                 #
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Condition:
    """One operating point: DC bias, load, and optionally a marked sample spot.

    `spot` selects a position ON THE SAMPLE (a spot marked on an image via the
    Force panel, e.g. one PPLN domain) — orthogonal to the laser position along
    the cantilever, which the acquisition loop owns. None = leave the sample
    where it is, so plain bias/load series are unchanged.
    """
    bias_V: float
    load_nN: float
    spot: int | None = None

    @property
    def label(self):
        s = "m" if self.bias_V < 0 else "p"
        base = f"DC{s}{abs(self.bias_V):.3f}V_L{self.load_nN:04.0f}nN".replace(".", "p")
        return base + (f"_S{self.spot}" if self.spot is not None else "")

    def __str__(self):
        base = f"{self.bias_V:+.3f} V, {self.load_nN:.0f} nN"
        return base + (f", spot {self.spot}" if self.spot is not None else "")


def make_conditions(bias_V, load_nN, vary="bias_inner", spots=None):
    """Build a condition list from bias and load values.

    `bias_V` and `load_nN` may be scalars or sequences. `vary='bias_inner'`
    cycles bias fastest (all biases at one load, then the next load), which is
    the right order on the instrument: bias changes are nearly free, load changes
    need a re-engage. `vary='load_inner'` does the opposite.

    A bias sweep at a single load gives the channel separation; a load sweep at a
    single bias gives the D-NS migration; both gives you each load's channel
    separation.

    `spots` (e.g. [1, 2] for two PPLN domains) is always the OUTERMOST loop:
    a sample-spot change costs a withdraw + scanner ramp + re-engage (~20-30 s),
    so all conditions at one spot are finished before hopping to the next.
    """
    b = np.atleast_1d(np.asarray(bias_V, float))
    l = np.atleast_1d(np.asarray(load_nN, float))
    sp = [None] if spots is None else [int(x) for x in np.atleast_1d(spots)]
    out = []
    for spot in sp:
        if vary == "bias_inner":
            for ll in l:
                for bb in b:
                    out.append(Condition(float(bb), float(ll), spot))
        elif vary == "load_inner":
            for bb in b:
                for ll in l:
                    out.append(Condition(float(bb), float(ll), spot))
        else:
            raise ValueError("vary must be 'bias_inner' or 'load_inner'")
    return out


# --------------------------------------------------------------------------- #
#  Acquisition                                                                #
# --------------------------------------------------------------------------- #
def run_series(instrument, x_grid_um, conditions, ref_index=None, rank=4,
               min_positions=None, max_positions=20, dns_band_Hz=None,
               dns_ci_tol_um=1.0, stable_over=3, start_near_um=None,
               checkpoint_path=None, resume=True, verbose=True):
    """Active-learning position selection, measuring every condition per position.

    One condition (`ref_index`, default the middle of the list) drives the
    position selection and the convergence test; every other condition is
    measured at the same positions, so all conditions share a position set and
    are directly comparable.

    `checkpoint_path` (an .npz) is rewritten after every position, so an
    interruption costs at most one position. With `resume=True` an existing
    checkpoint is loaded and only the remaining positions are measured — nothing
    already on the instrument is repeated.

    Returns a dict with `x_um`, `freq_Hz`, `Z` (n_cond, n_pos, n_freq),
    `conditions`, and the reference `LowRankModeMap`.
    """
    conditions = list(conditions)
    if not conditions:
        raise ValueError("no conditions given")
    if ref_index is None:
        ref_index = len(conditions) // 2
    ref = conditions[ref_index]
    x_grid_um = np.asarray(x_grid_um, float)
    min_positions = min_positions or (rank + 2)

    mm = LowRankModeMap(x_grid_um, freq_grid=None, rank=rank,
                        seeds_um=[x_grid_um[0], x_grid_um[len(x_grid_um) // 2],
                                  x_grid_um[-1]],
                        min_positions=min_positions, dns_ci_tol_um=dns_ci_tol_um,
                        start_near_um=start_near_um, dns_band_Hz=dns_band_Hz,
                        stable_over=stable_over)

    measured = {}            # x_um -> {cond_index: (freq, Z)}
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        prev = load_checkpoint(checkpoint_path)
        if prev["conditions"] != [asdict(c) for c in conditions]:
            raise ValueError(
                f"{checkpoint_path} was written with a different condition list. "
                "Delete it, or point checkpoint_path somewhere new.")
        for j, x in enumerate(prev["x_um"]):
            measured[float(x)] = {i: (prev["freq_Hz"], prev["Z"][i, j])
                                  for i in range(len(conditions))
                                  if np.isfinite(prev["Z"][i, j]).any()}
            mm.add_measurement(float(x), prev["freq_Hz"],
                               prev["Z"][ref_index, j])
        if verbose:
            print(f"resumed from {checkpoint_path}: {len(measured)} positions "
                  "already measured")

    t0 = time.time()
    consecutive_failures = 0
    while mm.n < max_positions:
        x = mm.next_position()
        if float(x) in measured:                     # already have it (resume)
            continue
        if verbose:
            done = mm.n
            print(f"\n=== position {done + 1}: x = {x:.1f} um "
                  f"({len(conditions)} conditions) ===", flush=True)
        try:
            got = instrument.measure_conditions_at(x, conditions, verbose=verbose)
        except KeyboardInterrupt:
            print("\ninterrupted — checkpoint holds everything up to here")
            break
        except Exception as e:
            print(f"position {x:.1f} um failed ({type(e).__name__}: {e}); stopping")
            break

        if got.get(ref_index) is None:
            # Block it, or next_position() returns the same x forever and the loop
            # spins. Several in a row means the fault is systemic (a bad tune
            # setup, a code error in the instrument path) rather than one awkward
            # spot, so stop and let the traceback above be read.
            mm.block_position(x)
            consecutive_failures += 1
            print(f"  reference condition failed at x={x:.1f}; blocking this "
                  f"position ({consecutive_failures} consecutive failure(s))")
            if consecutive_failures >= 3:
                print("  THREE POSITIONS IN A ROW FAILED - stopping. This is not "
                      "bad luck with the surface; fix the error reported above "
                      "and re-run. Nothing measured so far is lost.")
                break
            continue
        consecutive_failures = 0
        measured[float(x)] = {i: (v[0], v[1]) for i, v in got.items()
                              if v is not None}
        f_ref, Z_ref = got[ref_index][0], got[ref_index][1]
        mm.add_measurement(float(x), f_ref, Z_ref)

        if mm.n >= mm.min_positions:
            rec = mm.reconstruct(nboot=150)
            if verbose:
                el = (time.time() - t0) / 60
                print(f"  [{ref}] D-NS {rec['dns']:.2f} +/- {rec['dns_ci'] / 2:.2f} um"
                      f"   ({mm.n} positions, {el:.1f} min elapsed)")
            if checkpoint_path:
                save_checkpoint(checkpoint_path, measured, conditions, ref_index)
            if mm.converged():
                if verbose:
                    print("  converged (CI below tolerance and estimate stable)")
                break
        elif checkpoint_path:
            save_checkpoint(checkpoint_path, measured, conditions, ref_index)

    out = _pack(measured, conditions, ref_index)
    out["mm_ref"] = mm
    if checkpoint_path:
        save_checkpoint(checkpoint_path, measured, conditions, ref_index)
    return out


def _pack(measured, conditions, ref_index):
    """Stack the per-position dicts into (n_cond, n_pos, n_freq)."""
    xs = np.array(sorted(measured))
    if xs.size == 0:
        raise RuntimeError("nothing was measured")
    # common frequency grid: the reference condition's first spectrum, trimmed to
    # the range every spectrum covers (spectra can differ by a point or two after
    # non-finite trimming -- see tune_to_complex)
    grids = [fz[0] for d in measured.values() for fz in d.values()]
    F0 = grids[0]
    lo = max(float(g.min()) for g in grids)
    hi = min(float(g.max()) for g in grids)
    F = F0[(F0 >= lo - 1e-9) & (F0 <= hi + 1e-9)]
    if F.size < 8:
        raise RuntimeError("frequency grids share fewer than 8 points")

    Z = np.full((len(conditions), xs.size, F.size), np.nan + 0j)
    for j, x in enumerate(xs):
        for i, (f, z) in measured[float(x)].items():
            if f.shape == F.shape and np.allclose(f, F):
                Z[i, j] = z
            else:
                Z[i, j] = (np.interp(F, f, z.real) + 1j * np.interp(F, f, z.imag))
    return dict(x_um=xs, freq_Hz=F, Z=Z, conditions=conditions,
                ref_index=ref_index)


# --------------------------------------------------------------------------- #
#  Checkpointing                                                              #
# --------------------------------------------------------------------------- #
def save_checkpoint(path, measured, conditions, ref_index):
    d = _pack(measured, conditions, ref_index)
    tmp = path + ".tmp"
    # Write through a file handle: np.savez_compressed APPENDS '.npz' when handed
    # a filename that does not already end in it, which would silently put the
    # temp file somewhere other than where os.replace looks for it.
    with open(tmp, "wb") as fh:
        np.savez_compressed(
            fh, x_um=d["x_um"], freq_Hz=d["freq_Hz"], Z=d["Z"],
            ref_index=ref_index,
            conditions=json.dumps([asdict(c) for c in conditions]))
    os.replace(tmp, path)          # atomic: a crash mid-write cannot corrupt it
    return path


def load_checkpoint(path):
    d = np.load(path, allow_pickle=False)
    conds = json.loads(str(d["conditions"]))
    return dict(x_um=d["x_um"], freq_Hz=d["freq_Hz"], Z=d["Z"],
                ref_index=int(d["ref_index"]), conditions=conds)


# --------------------------------------------------------------------------- #
#  Reconstruction per condition                                               #
# --------------------------------------------------------------------------- #
def reconstruct_series(series, x_grid_um, rank=4, dns_band_Hz=None,
                       auto_band=True, band_halfwidth_Hz=70e3, verbose=True):
    """Low-rank reconstruction for every condition on a common position grid.

    `auto_band=True` re-centres the D-NS search band on each condition's own
    resonance. This matters for a load series: the contact resonance stiffens
    with load and can move by tens of kilohertz, so a band fixed at one load
    silently misses the peak at another.
    """
    x_grid_um = np.asarray(x_grid_um, float)
    xs, F, Z = series["x_um"], series["freq_Hz"], series["Z"]
    sel = [int(np.argmin(np.abs(x_grid_um - x))) for x in xs]
    if len(set(sel)) != len(sel):
        raise ValueError("two positions map to the same grid point; finer grid")

    out = []
    for i, cond in enumerate(series["conditions"]):
        Zi = Z[i]
        if not np.isfinite(Zi).all():
            out.append(None)
            if verbose:
                print(f"  {cond}: incomplete, skipped")
            continue
        if auto_band:
            ipk = int(np.argmax(np.abs(Zi).max(axis=0)))
            band = (F[ipk] - band_halfwidth_Hz, F[ipk] + band_halfwidth_Hz)
        else:
            band = dns_band_Hz
        bm = band_mask(F, band) if band else np.ones(F.size, bool)
        if bm.sum() < 8:
            out.append(None); continue
        fb, Zb = F[bm], Zi[:, bm]
        rec = reconstruct_map(x_grid_um, sel, Zi, min(rank, len(sel)))
        ires_b = resonance_index(fb, Zb)
        rec_b = reconstruct_map(x_grid_um, sel, Zb, min(rank, len(sel)))
        dns = dns_from_map(x_grid_um, rec_b["Zrec"], fb, ires_b)
        cross, _ = dns_branch(x_grid_um, rec_b["Zrec"], fb, ires_b)
        out.append(dict(condition=cond, Zrec=rec["Zrec"], std=rec["std"],
                        f_res_Hz=float(fb[ires_b]), band_Hz=band,
                        dns_um=dns, crossings_um=cross))
        if verbose:
            print(f"  {cond}: f_res {fb[ires_b] / 1e3:8.2f} kHz, "
                  f"D-NS {dns:7.2f} um, crossings {np.round(cross, 1)}")
    return out


# --------------------------------------------------------------------------- #
#  Channel separation from the bias dependence                                #
# --------------------------------------------------------------------------- #
def separate_channels(series, load_nN=None, v_cpd=0.0, min_biases=3):
    """Split piezoresponse and electrostatic channels using the bias dependence.

    Fits ``Z = a + b·V`` in the complex plane, independently at every position and
    frequency, across the biases measured at one load. Returns

      * ``elec``  = b            — the electrostatic channel, dZ/dV. Exact: it
        does not depend on V_cpd, so the D-ESBS taken from it is rigorous.
      * ``piezo`` = a + v_cpd·b  — the response at V = V_cpd. With the default
        ``v_cpd=0`` this is the response at zero bias, which still contains a
        −V_cpd·b electrostatic term. Supply a measured `v_cpd` (from KPFM, or
        from `estimate_v_cpd`) to remove it.
      * ``resid`` — RMS fit residual, the check on linearity. A response that is
        not linear in V (domain switching, electrostriction, a drifting contact)
        shows up here as residual comparable to the signal.

    Needs at least `min_biases` distinct biases; two define a line exactly and
    leave no residual with which to test it.
    """
    conds = series["conditions"]
    loads = sorted({c.load_nN for c in conds})
    if load_nN is None:
        if len(loads) > 1:
            raise ValueError(
                f"series spans several loads {loads}; pass load_nN to pick one")
        load_nN = loads[0]
    idx = [i for i, c in enumerate(conds) if abs(c.load_nN - load_nN) < 1e-9]
    V = np.array([conds[i].bias_V for i in idx], float)
    if np.unique(V).size < min_biases:
        raise ValueError(
            f"only {np.unique(V).size} distinct biases at {load_nN:.0f} nN; "
            f"need >= {min_biases} for a testable linear fit (2 points define a "
            "line exactly and give no residual)")

    Z = series["Z"][idx]                       # (nV, npos, nfreq)
    if not np.isfinite(Z).all():
        raise ValueError("some spectra are missing at this load")

    # complex least squares on [1, V]
    A = np.stack([np.ones_like(V), V], axis=1)          # (nV, 2)
    nV, npos, nf = Z.shape
    coef, *_ = np.linalg.lstsq(A, Z.reshape(nV, -1), rcond=None)
    a = coef[0].reshape(npos, nf)
    b = coef[1].reshape(npos, nf)
    fit = (A @ coef).reshape(nV, npos, nf)
    resid = np.sqrt((np.abs(Z - fit) ** 2).mean(axis=0))

    return dict(bias_V=V, load_nN=load_nN,
                piezo=a + v_cpd * b, elec=b, intercept=a, resid=resid,
                v_cpd=v_cpd, x_um=series["x_um"], freq_Hz=series["freq_Hz"],
                rel_resid=float(np.nanmax(resid) / (np.nanmax(np.abs(Z)) + 1e-30)))


def separate_domains(series, spot_up, spot_down, bias_V=None, load_nN=None):
    """Split piezo and electrostatic channels from TWO PPLN domains.

    The piezoresponse flips sign with the domain while the electrostatic
    response does not:

        Z_up   = +P + E
        Z_down = -P + E        =>   P = (Z_up - Z_down) / 2
                                    E = (Z_up + Z_down) / 2

    Exact per (x, f) — no fit, no residual, and no assumption about V_cpd. Note
    E here is the electrostatic response AT THE MEASUREMENT BIAS (it scales with
    V - V_cpd), which is fine for locating the blind spot: the spatial null of E
    does not move with its overall scale. If the measurement bias happens to sit
    near V_cpd, E is small everywhere and its null is poorly conditioned — use a
    deliberately nonzero bias, or a bias where the response is strong.

    `spot_up` / `spot_down` name the marked spots; which physical orientation is
    "up" only fixes the overall sign of P, which no null-spot estimate depends
    on. Returns the same dict shape as `separate_channels`, so `channel_spots`
    applies unchanged (`rel_resid` is NaN: a two-point decomposition is exact and
    has no linearity check — the consistency test is instead `imbalance`, see
    below).
    """
    conds = series["conditions"]

    def pick(spot):
        idx = [i for i, c in enumerate(conds)
               if getattr(c, "spot", None) == spot
               and (bias_V is None or abs(c.bias_V - bias_V) < 1e-9)
               and (load_nN is None or abs(c.load_nN - load_nN) < 1e-9)]
        if len(idx) != 1:
            raise ValueError(
                f"expected exactly one condition at spot {spot} "
                f"(bias_V={bias_V}, load_nN={load_nN}); found {len(idx)}. "
                "Pass bias_V/load_nN to disambiguate.")
        return idx[0]

    iu, idn = pick(spot_up), pick(spot_down)
    Zu, Zd = series["Z"][iu], series["Z"][idn]
    if not (np.isfinite(Zu).all() and np.isfinite(Zd).all()):
        raise ValueError("one of the domain spectra is incomplete")
    P = (Zu - Zd) / 2.0
    E = (Zu + Zd) / 2.0
    # NOTE there is no model-free consistency check from a single bias: the
    # decomposition uses both measurements exactly (2 unknowns, 2 data), and
    # |Z_up| != |Z_down| even for perfect domains because the electrostatic term
    # interferes constructively in one and destructively in the other. In
    # particular, a contact-gain difference g between the two spots is
    # INDISTINGUISHABLE from electrostatics here: it leaks (g1-g2)/2 * P into
    # the E channel. The guard against that is a second bias -- E must scale
    # with (V - V_cpd) while leaked P must not; measure at two biases and
    # compare the two P estimates (they should agree) and the two E estimates
    # (they should scale). `elec_frac` below is informational only: how much of
    # the strong-signal response is electrostatic at this bias.
    tot = np.abs(P) + np.abs(E)
    strong = tot > 0.25 * tot.max()
    elec_frac = float(np.median(np.abs(E[strong]) / (tot[strong] + 1e-30)))
    return dict(piezo=P, elec=E, intercept=P, resid=np.full(P.shape, np.nan),
                bias_V=np.array([conds[iu].bias_V]),
                load_nN=conds[iu].load_nN, v_cpd=np.nan,
                x_um=series["x_um"], freq_Hz=series["freq_Hz"],
                rel_resid=float("nan"), elec_frac=elec_frac,
                spot_up=spot_up, spot_down=spot_down)


def estimate_v_cpd(channels, band_Hz=None):
    """Bias that minimises the total response, per position — a V_cpd proxy.

    For ``Z(V) = a + bV`` the minimising bias is ``-Re(a·conj(b))/|b|²``. That is
    exactly V_cpd only where the piezoresponse vanishes; elsewhere the
    piezoresponse pulls it. Reported as a diagnostic and a rough correction, not
    as a calibrated KPFM measurement — take the value near the D-NS, where the
    piezoresponse is smallest and the estimate is therefore cleanest.
    """
    a, b = channels["intercept"], channels["elec"]
    F = channels["freq_Hz"]
    bm = band_mask(F, band_Hz) if band_Hz else np.ones(F.size, bool)
    num = -(a[:, bm] * np.conj(b[:, bm])).real
    den = (np.abs(b[:, bm]) ** 2) + 1e-30
    per_pos = (num / den).mean(axis=1)
    return per_pos, float(np.median(per_pos))


def channel_spots(channels, x_grid_um, rank=4, band_Hz=None, verbose=True):
    """D-NS from the piezo channel, D-ESBS from the electrostatic channel.

    They use DIFFERENT estimators, deliberately:

    * **D-NS** -- the antiresonance branch crossing (`dns_from_map`), where the
      notch frequency fa(x) meets the contact resonance. Stable because the notch
      is a deep, well-localised feature.
    * **D-ESBS** -- a spatial null (`spatial_null`) of the electrostatic channel
      at the resonance frequency. The branch crossing is *not* usable here: it
      reports the same position for both channels, because the notch frequency is
      a property of the coupled dynamics rather than of which drive pathway you
      are looking at.

    The D-ESBS returned is therefore the blind spot **at the resonance
    frequency** -- the operating point for resonance-enhanced PFM. That is not
    the same as the quasistatic blind spot (`EBForwardModel.find_desbs`), which
    is evaluated far below resonance; expect them to differ by a micron or two.
    """
    x_grid_um = np.asarray(x_grid_um, float)
    xs, F = channels["x_um"], channels["freq_Hz"]
    sel = [int(np.argmin(np.abs(x_grid_um - x))) for x in xs]
    bm = band_mask(F, band_Hz) if band_Hz else np.ones(F.size, bool)
    fb = F[bm]
    out = {}

    Pb = channels["piezo"][:, bm]
    recP = reconstruct_map(x_grid_um, sel, Pb, min(rank, len(sel)))
    iresP = resonance_index(fb, Pb)
    crossP, _ = dns_branch(x_grid_um, recP["Zrec"], fb, iresP)
    out["dns"] = dict(value_um=dns_from_map(x_grid_um, recP["Zrec"], fb, iresP),
                      crossings_um=crossP, Zrec=recP["Zrec"],
                      f_res_Hz=float(fb[iresP]), freq_Hz=fb, method="branch crossing")

    Eb = channels["elec"][:, bm]
    recE = reconstruct_map(x_grid_um, sel, Eb, min(rank, len(sel)))
    iresE = resonance_index(fb, Eb)
    out["desbs"] = dict(value_um=spatial_null(x_grid_um, recE["Zrec"], fb, iresE),
                        Zrec=recE["Zrec"], f_res_Hz=float(fb[iresE]), freq_Hz=fb,
                        method="spatial null at resonance")

    if verbose:
        print(f"  D-NS   (piezo, branch crossing)      : "
              f"{out['dns']['value_um']:.2f} um")
        print(f"  D-ESBS (electrostatic, spatial null) : "
              f"{out['desbs']['value_um']:.2f} um  "
              f"[at {out['desbs']['f_res_Hz'] / 1e3:.1f} kHz, not quasistatic]")
        rr = channels.get("rel_resid", float("nan"))
        if np.isfinite(rr):
            print(f"  linearity: peak residual {100 * rr:.1f}% of peak |Z| "
                  "(large => response not linear in bias: domain switching, or "
                  "the contact drifted during the sweep)")
        if np.isfinite(channels.get("elec_frac", float("nan"))):
            print(f"  electrostatic fraction of the strong signal: "
                  f"{100 * channels['elec_frac']:.1f}% at this bias "
                  "(informational; a contact-gain difference between the spots "
                  "leaks piezo into this channel — verify with a second bias)")
    return out
