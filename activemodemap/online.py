"""Instrument abstraction for the online active-learning loop.

The loop is hardware-agnostic: it only needs an object with

    measure_at(x_um) -> (freq_Hz, Z_complex, meta)

that positions the detection spot at x (µm) and returns one complex spectrum.
Two implementations are provided:

  * VirtualInstrument — simulates the measurement from the physics forward model,
    so the notebook logic can be dry-run without the microscope.
  * (AsylumInstrument lives in `asylum.py`; it wraps the Igor/Asylum automation.)
"""

from __future__ import annotations

import numpy as np


class Instrument:
    """Base class. Subclass and implement measure_at()."""

    def measure_at(self, x_um):
        raise NotImplementedError

    def close(self):
        pass


class VirtualInstrument(Instrument):
    """Simulated instrument backed by the EB forward model.

    Returns complex spectra on a fixed frequency grid with realistic noise and
    a finite detection-spot footprint, matching the real instrument's interface
    so the same loop code runs in both settings.
    """

    def __init__(self, freq_grid_Hz, theta_true=None, noise_floor=0.08,
                 spot_fwhm_um=4.0, domain_sign=+1.0, rng=None,
                 geom=None):
        from .forward_model import EBForwardModel, ProbeGeometry
        self.rng = rng or np.random.default_rng(0)
        self.freq = np.asarray(freq_grid_Hz, float)
        geom = geom or ProbeGeometry()
        # build a model whose frequency grid matches the requested window
        fs = geom.f0_hz / (1.87510407 ** 2)             # Hz per scaled omega
        omega = self.freq / fs
        m = EBForwardModel(geom=geom, n_modes=14, nx=241,
                           nf=self.freq.size,
                           omega_lo=float(omega.min()),
                           omega_hi=float(omega.max()))
        self.model = m
        self.L_um = geom.L_um
        self.domain_sign = domain_sign
        if theta_true is None:
            theta_true = np.array([3.2, np.log10(2000 / 3), 1.4, -0.2, 0.05])
        self.theta_true = np.asarray(theta_true, float)
        resp = m.response(self.theta_true)
        self._map = m.measured(self.theta_true, domain_sign, resp)   # (nf, nx)
        self.scale_qs = np.abs(self._map[0]).max()
        self.sigma = noise_floor * self.scale_qs
        # detection-spot blur along position
        sig_um = spot_fwhm_um / 2.355
        self._sig_pts = sig_um / (self.L_um * (m.xi[1] - m.xi[0]))
        if self._sig_pts > 0.3:
            from scipy.ndimage import gaussian_filter1d
            self._map = (gaussian_filter1d(self._map.real, self._sig_pts, axis=1)
                         + 1j * gaussian_filter1d(self._map.imag, self._sig_pts, axis=1))
        # true D-NS (distance from free end) for reference
        self.dns_um_from_end = (1 - m.find_dns(resp)) * self.L_um
        self.n_measurements = 0

    def measure_at(self, x_um):
        """x_um measured from the base end of the accessible span (0..L)."""
        xi = np.clip(x_um / self.L_um, 0, 1)
        i = int(np.argmin(np.abs(self.model.xi - xi)))
        clean = self._map[:, i]
        noise = self.sigma * (self.rng.standard_normal(clean.size)
                              + 1j * self.rng.standard_normal(clean.size)) / np.sqrt(2)
        self.n_measurements += 1
        meta = {"xi": self.model.xi[i], "position_index": i}
        return self.freq.copy(), clean + noise, meta


# --------------------------------------------------------------------------- #
#  Live plotting                                                              #
# --------------------------------------------------------------------------- #
def _draw_state(fig, ax, mm, rec, title=None, truth=None):
    """Render the current reconstruction into a 2x2 axes array (used by both
    `plot_state` and `LiveModeMapPlot`).

    Shows the map over the WHOLE tune window, not just the D-NS band: the
    low-rank fit is an independent solve per frequency, so the full spectrum is
    reconstructed at every position and there is no reason to hide it. The band
    used to locate the null is outlined instead.
    """
    import numpy as np
    x, f = mm.x_grid, mm.freq / 1e3
    ext = [x[0], x[-1], f[0], f[-1]]
    amp = np.abs(rec["Zrec"]).T                      # (nfreq, npos)
    # Every reduction below is NaN-safe. A single NaN in Zrec would otherwise
    # make .max() NaN, which blanks the uncertainty panel entirely and leaves the
    # map painting only part of its axes -- silent, and easy to misread as a
    # plotting bug rather than bad data.
    amp_max = np.nanmax(amp) if np.isfinite(amp).any() else 1.0
    n_bad = int((~np.isfinite(rec["Zrec"])).sum())
    for a in ax.ravel():
        a.clear()

    # -- 1. reconstructed map, full spectrum ------------------------------- #
    a = ax[0, 0]
    a.imshow(np.log10(amp + 1e-12), origin="lower", aspect="auto",
             extent=ext, cmap="viridis")
    for xs in rec["x_sel"]:
        a.axvline(xs, color="w", lw=0.6, alpha=0.75)
    if np.isfinite(rec["dns"]):
        a.axvline(rec["dns"], color="#e34948", ls="--", lw=1.5)
    if rec.get("dns_band_Hz"):
        lo, hi = rec["dns_band_Hz"]
        a.axhline(lo / 1e3, color="#ffd166", lw=1.0, ls=":")
        a.axhline(hi / 1e3, color="#ffd166", lw=1.0, ls=":")
    ttl = title or f"reconstruction, full tune ({mm.n} positions)"
    if n_bad:
        ttl += f"  [{n_bad} non-finite cells]"
    a.set_title(ttl, fontsize=10)
    a.set_xlabel("position (µm)"); a.set_ylabel("frequency (kHz)")

    # -- 2. relative uncertainty ------------------------------------------- #
    a = ax[0, 1]
    rel = (rec["std"] / (amp_max + 1e-12)).T
    if np.isfinite(rel).any():
        a.imshow(rel, origin="lower", aspect="auto", extent=ext, cmap="magma",
                 vmin=float(np.nanmin(rel)), vmax=float(np.nanmax(rel)))
    else:
        a.text(0.5, 0.5, "uncertainty all non-finite", ha="center", va="center",
               transform=a.transAxes, fontsize=9)
    a.set_title("uncertainty (1σ, rel.)", fontsize=10)
    a.set_xlabel("position (µm)"); a.set_ylabel("frequency (kHz)")

    # -- 3. spectra: measured vs reconstructed, whole window --------------- #
    a = ax[1, 0]
    cmap = _plt().cm.viridis
    for k, i in enumerate(rec["sel_idx"]):
        c = cmap(k / max(len(rec["sel_idx"]) - 1, 1))
        a.semilogy(f, np.abs(rec["Zsel"][k]), lw=0.7, color=c, alpha=0.55)
        a.semilogy(f, np.abs(rec["Zrec"][i]), lw=1.3, color=c, ls="--")
    if rec.get("dns_band_Hz"):
        a.axvspan(rec["dns_band_Hz"][0] / 1e3, rec["dns_band_Hz"][1] / 1e3,
                  color="#ffd166", alpha=0.16)
    a.set_xlim(f[0], f[-1])          # full window, not just the finite part
    a.set_title("spectra: measured (thin) vs fit (dashed)", fontsize=10)
    a.set_xlabel("frequency (kHz)"); a.set_ylabel("|Z|")

    # -- 4. D-NS convergence ----------------------------------------------- #
    a = ax[1, 1]
    h = getattr(mm, "history", [])
    if h:
        n = [e["n"] for e in h]
        d = np.array([e["dns"] for e in h], float)
        ci = np.array([e["dns_ci"] for e in h], float)
        a.plot(n, d, "o-", color="#2a78d6", ms=4, lw=1.3)
        ok = np.isfinite(ci)
        if ok.any():
            a.fill_between(np.array(n)[ok], (d - ci / 2)[ok], (d + ci / 2)[ok],
                           color="#2a78d6", alpha=0.2)
        if truth is not None:
            a.axhline(truth, color="k", ls=":", lw=1.2, label=f"dense {truth:.1f} µm")
            a.legend(fontsize=7.5, frameon=False)
        a.set_xlabel("positions measured"); a.set_ylabel("D-NS (µm)")
        ttl = f"D-NS = {rec['dns']:.2f} ± {rec['dns_ci'] / 2:.2f} µm"
        if len(rec.get("crossings", [])) > 1:
            ttl += "  (multiple nulls!)"
        a.set_title(ttl, fontsize=10)
    else:
        a.axis("off")
    return ax


def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_state(mm, rec, ax=None, title=None, truth=None):
    """Draw the current reconstruction over the full tune window.

    `mm` is a LowRankModeMap, `rec` its latest reconstruct() output. Returns the
    2x2 axes array. For a loop that redraws every position, prefer
    `LiveModeMapPlot`, which reuses one figure instead of leaking a new one per
    iteration.
    """
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(2, 2, figsize=(12, 7.2))
    else:
        fig = ax.ravel()[0].figure
    _draw_state(fig, ax, mm, rec, title=title, truth=truth)
    fig.tight_layout()
    return ax


class LiveModeMapPlot:
    """One reusable figure that updates in place as the loop measures.

    Usage inside the acquisition loop::

        live = LiveModeMapPlot()
        ...
        rec = mm.reconstruct()
        live.update(mm, rec)

    Each `update` redraws into the same figure and replaces the previous output
    cell, so the notebook shows a single panel that evolves rather than a growing
    stack of plots. Redraw takes ~0.2 s, against ~60 s per position on the
    instrument, so it is free in practice. With the `ipympl` backend
    (`%matplotlib widget`) the canvas updates without the flicker of the inline
    backend, but the inline default works fine.
    """

    def __init__(self, figsize=(12, 7.2)):
        plt = _plt()
        self.fig, self.ax = plt.subplots(2, 2, figsize=figsize)
        self._shown = False

    def update(self, mm, rec, title=None, truth=None):
        _draw_state(self.fig, self.ax, mm, rec, title=title, truth=truth)
        self.fig.tight_layout()
        try:
            from IPython.display import display, clear_output
            clear_output(wait=True)
            display(self.fig)
            self._shown = True
        except Exception:
            pass
        return self.ax

    def close(self):
        _plt().close(self.fig)


# --------------------------------------------------------------------------- #
#  Dense ground-truth acquisition                                             #
# --------------------------------------------------------------------------- #
def dense_grid(x_start_um, x_stop_um, step_um):
    """Uniformly spaced positions from `x_start_um` to `x_stop_um` inclusive.

    Order is preserved: pass the position the laser is parked at as
    `x_start_um` and the sweep walks away from it in one direction at a constant
    pitch, which is both what you want physically (no wasted traverse, no
    direction reversals to accumulate backlash) and what makes the reference
    comparable position to position.

    `x_stop_um` is included exactly even when the span is not a whole multiple of
    `step_um`, so the sweep never overshoots past the end you declared.
    """
    import numpy as np
    x0, x1, st = float(x_start_um), float(x_stop_um), abs(float(step_um))
    if st <= 0:
        raise ValueError("step_um must be positive")
    n = int(np.floor(abs(x1 - x0) / st))
    sgn = 1.0 if x1 >= x0 else -1.0
    pts = [x0 + sgn * st * k for k in range(n + 1)]
    if abs(pts[-1] - x1) > 1e-9:
        pts.append(x1)                       # land exactly on the declared end
    return np.asarray(pts, float)


def dense_reference_sweep(instrument, x_um, verbose=True, enforce_limits=True,
                          min_overlap_frac=0.8):
    """Measure every position in `x_um` — the dense ground truth.

    No reconstruction and no low-rank assumption: this *is* the map, sampled at
    whatever spacing you pass. Returns ``(x_um, freq_Hz, Z (npos, nfreq))`` sorted
    by position, but positions are MEASURED in the order given — pass them
    starting from wherever the laser is parked.

    Frequency grids are harmonised at the end rather than required to match
    exactly. Two different things make them differ, and they need different
    responses:

    * **Non-finite trimming.** Igor leaves Amp/Phase NaN outside the swept range
      and `tune_to_complex` drops those points, so the surviving grid can differ
      by a point or two between positions. Harmless — resampling onto the common
      grid is the right answer.
    * **Auto-recentring.** If the tune is tracking the resonance, the window
      genuinely moves between positions and the map is not comparable
      position-to-position. That must be surfaced, not silently interpolated
      away, so this prints the per-position windows and warns when the mutual
      overlap drops below `min_overlap_frac` of the first window.

    Never discards measurements: on any problem it returns what it managed to
    collect. Interrupting with the kernel is likewise safe.

    Budget roughly a minute per position on the instrument.
    """
    import numpy as np
    import time
    x_um = np.asarray(x_um, float)
    lim = getattr(instrument, "x_limits_um", None)
    if enforce_limits and lim is not None:
        inside = (x_um >= lim[0] - 1e-9) & (x_um <= lim[1] + 1e-9)
        if not inside.all():
            drop = x_um[~inside]
            print(f"  refusing {drop.size} position(s) outside the reachable "
                  f"range [{lim[0]:.1f}, {lim[1]:.1f}] um: "
                  f"{np.round(drop, 1)} -- these would drive the spot off the "
                  "cantilever. Adjust x_limits_um if the range is wrong.")
            x_um = x_um[inside]
        if x_um.size == 0:
            raise ValueError("no positions left inside the reachable range")

    xs, raw = [], []          # raw = [(freq, Z)] at native grids
    t0 = time.time()
    for k, x in enumerate(x_um):
        try:
            f, Z, _ = instrument.measure_at(float(x))
        except KeyboardInterrupt:
            print(f"\ninterrupted after {len(xs)} positions — keeping those")
            break
        except Exception as e:
            print(f"\nposition {x:.1f} um failed ({type(e).__name__}: {e}); "
                  f"keeping the {len(xs)} positions measured so far")
            break
        xs.append(float(x))
        raw.append((np.asarray(f, float), np.asarray(Z, complex)))
        if verbose:
            el = time.time() - t0
            eta = (el / (k + 1)) * (len(x_um) - k - 1) / 60.0
            print(f"  [{k + 1}/{len(x_um)}] x = {x:7.1f} um   "
                  f"{f.size} pts {f[0] / 1e3:.1f}-{f[-1] / 1e3:.1f} kHz   "
                  f"elapsed {el / 60:.1f} min, ~{eta:.0f} min left", flush=True)

    if not xs:
        raise RuntimeError("no positions were measured")

    # ---- harmonise the frequency grids -------------------------------------
    sizes = {r[0].size for r in raw}
    lo = max(float(r[0].min()) for r in raw)
    hi = min(float(r[0].max()) for r in raw)
    F0 = raw[0][0]
    span0 = float(F0[-1] - F0[0])
    overlap = (hi - lo) / span0 if span0 > 0 else 0.0

    # Edge drift is the sensitive test for recentring: a window that slides
    # steadily can still show a high mutual overlap, so overlap alone misses it.
    edge_drift = max(max(abs(float(r[0][0]) - float(F0[0])),
                         abs(float(r[0][-1]) - float(F0[-1]))) for r in raw)
    drifting = span0 > 0 and (edge_drift / span0) > 0.02

    if len(sizes) > 1 or overlap < 1.0 - 1e-9:
        print(f"\n  frequency grids differ between positions "
              f"(point counts {sorted(sizes)}); mutual overlap "
              f"{lo / 1e3:.1f}-{hi / 1e3:.1f} kHz = {100 * overlap:.1f}% of the "
              "first window. Resampling onto the common grid.")
        print(f"  window edges drift by up to {edge_drift / 1e3:.2f} kHz "
              f"({100 * edge_drift / span0:.1f}% of the span)")
        if overlap < min_overlap_frac or drifting:
            print("  ** WARNING: the windows barely overlap. That is not NaN "
                  "trimming, it means the tune is RECENTRING between positions. "
                  "Turn resonance tracking OFF in Igor and re-run -- this map is "
                  "not comparable position to position. **")
            for xx, (ff, _) in zip(xs, raw):
                print(f"     x={xx:7.1f} um: {ff.size:5d} pts, "
                      f"{ff[0] / 1e3:9.2f}-{ff[-1] / 1e3:9.2f} kHz")

    common = F0[(F0 >= lo - 1e-9) & (F0 <= hi + 1e-9)]
    if common.size < 8:
        raise RuntimeError(
            f"only {common.size} frequency points are common to all positions "
            f"({lo / 1e3:.1f}-{hi / 1e3:.1f} kHz). The tune window moved too much; "
            "disable resonance tracking in Igor.")
    Zs = []
    for ff, ZZ in raw:
        if ff.shape == common.shape and np.allclose(ff, common):
            Zs.append(ZZ)
        else:
            Zs.append(np.interp(common, ff, ZZ.real)
                      + 1j * np.interp(common, ff, ZZ.imag))
    o = np.argsort(xs)
    return np.asarray(xs)[o], common, np.asarray(Zs)[o]


# --------------------------------------------------------------------------- #
#  Position-scale correction                                                  #
# --------------------------------------------------------------------------- #
class PositionScale:
    """Affine repair of a position axis: ``x_true = scale * x_logged + offset``.

    `DoLDMove` under-delivers on small steps, so a sweep made of many short
    relative moves travels less than its axis says. On Demo6Fullgrid a dense
    pass of 280 chained 0.5 µm steps covered only ~85% of its commanded 140 µm,
    which put its labelled x = 85 µm about 25 µm away from the sparse pass's
    x = 85 µm. Comparing the two on their logged axes is therefore comparing
    different places on the cantilever.

    `scale == 1.0` and `offset == 0.0` is the identity, i.e. "no correction" —
    use `PositionScale.identity()` to say that explicitly.

    Attributes other than scale/offset are fit diagnostics and are None for a
    hand-built instance.
    """

    def __init__(self, scale=1.0, offset=0.0, n_used=None, resid_um=None,
                 excluded_x_um=None, matched=None, source="manual"):
        self.scale = float(scale)
        self.offset = float(offset)
        self.n_used = n_used
        self.resid_um = resid_um
        self.excluded_x_um = excluded_x_um
        self.matched = matched
        self.source = source

    @classmethod
    def identity(cls):
        return cls(1.0, 0.0, source="identity")

    @classmethod
    def from_scale(cls, scale, anchor_um):
        """A travel shortfall of `scale`, accumulated away from `anchor_um`.

        This — not `offset=0` — is the physically meaningful one-parameter form.
        A sweep is parked somewhere, usually the free end, and the shortfall
        builds up as it steps away, so the axis is correct at the anchor and
        wrong in proportion to the distance travelled from it. Writing
        ``x_true = scale * x`` instead pivots about x = 0, i.e. about the
        clamped base, which no sweep is anchored to: with scale 0.85 that moves
        a correct 225 µm point by 34 µm and is simply a different (wrong) claim.
        """
        anchor_um = float(anchor_um)
        return cls(scale, anchor_um * (1.0 - float(scale)),
                   source=f"scale about {anchor_um:.1f}um")

    @property
    def is_identity(self):
        return self.scale == 1.0 and self.offset == 0.0

    @property
    def rms_um(self):
        if self.resid_um is None or len(self.resid_um) == 0:
            return None
        return float(np.sqrt(np.mean(np.asarray(self.resid_um) ** 2)))

    def apply(self, x_um):
        """Logged positions -> corrected positions."""
        return self.scale * np.asarray(x_um, float) + self.offset

    def invert(self, x_true_um):
        """Corrected positions -> the logged positions that produced them."""
        return (np.asarray(x_true_um, float) - self.offset) / self.scale

    def __repr__(self):
        r = self.rms_um
        tail = "" if r is None else f", n={self.n_used}, rms={r:.2f}um"
        return (f"PositionScale(scale={self.scale:.4f}, "
                f"offset={self.offset:+.2f}um, source={self.source}{tail})")


def _xv(records, x_key="position_x_um", invols_key="invols_m_per_V"):
    """(x, invols) arrays from a record list, a DataFrame, or a 2-tuple."""
    if isinstance(records, (tuple, list)) and len(records) == 2 \
            and np.ndim(records[0]) == 1 and not isinstance(records[0], dict):
        x, v = records
    elif hasattr(records, "columns"):                       # DataFrame
        x, v = records[x_key].values, records[invols_key].values
    else:                                                   # sequence of dicts
        rs = [r for r in records if r is not None]
        x = [r[x_key] for r in rs]
        v = [r[invols_key] for r in rs]
    x = np.asarray(x, float)
    v = np.asarray(v, float)
    good = np.isfinite(x) & np.isfinite(v) & (v > 0)
    return x[good], v[good]


def fit_position_scale(sparse_records, dense_records, smooth=9, min_points=3,
                       x_key="position_x_um", invols_key="invols_m_per_V",
                       verbose=True):
    """Recover a dense sweep's true position axis from its own InvOLS log.

    The optical-lever sensitivity is a steep, monotonic function of where the
    spot actually sits on the beam — a factor ~5 from free end to base on a
    Multi75E-G — and `AsylumInstrument` already measures it at every position of
    every pass. That makes InvOLS a position ruler that costs nothing extra:
    a pass built from a few large moves is the trustworthy reference, and the
    scale factor that maps the dense pass's InvOLS-vs-position curve onto it is
    the travel shortfall.

    Method: invert the (monotonised) dense log-InvOLS curve to find, for each
    sparse point, the dense LABEL carrying the same sensitivity, then regress
    the sparse labels on those matched dense labels. Regressing rather than
    averaging per-point ratios keeps the support fixed as the fit moves, and
    hands back an offset as well as a scale.

    Sparse points whose InvOLS lies outside the dense pass's InvOLS range are
    outside the dense pass's true coverage and are excluded — on Demo6 that
    correctly drops x = 85 and 85.5 µm, which the dense sweep never reached.

    Validated on Demo6Fullgrid: scale 0.8466 with 0.65 µm rms residuals over 7
    points, against 0.8505 ± 0.0031 measured independently from the FAMap
    optical images — 0.5% agreement, and stable over smooth = 1..25.

    Parameters
    ----------
    sparse_records, dense_records
        Record lists (`AsylumInstrument.records`), DataFrames of the saved log
        CSVs, or `(x_um, invols_m_per_V)` array pairs. The FIRST argument is the
        reference frame — pass the pass made of few large moves.
    smooth : int
        Running-median width applied to dense log-InvOLS before monotonising.
    min_points : int
        Refuse to fit below this many usable sparse points.

    Returns
    -------
    PositionScale

    Raises
    ------
    ValueError
        If fewer than `min_points` sparse points can be matched. That means the
        two passes' InvOLS ranges barely overlap, which is itself a red flag —
        do not paper over it with a fabricated scale.
    """
    xs, vs = _xv(sparse_records, x_key, invols_key)
    xd, vd = _xv(dense_records, x_key, invols_key)
    if xs.size < min_points or xd.size < 3:
        raise ValueError(
            f"need >= {min_points} sparse and >= 3 dense InvOLS records, "
            f"got {xs.size} and {xd.size}")

    o = np.argsort(xd)
    X, V = xd[o], np.log(vd[o])
    if smooth and smooth > 1:                     # kill per-point measurement noise
        k = int(smooth) // 2
        V = np.array([np.median(V[max(0, i - k):i + k + 1]) for i in range(V.size)])
    V = np.minimum.accumulate(V)                  # InvOLS falls toward the free end
    Xi, Vi = X[::-1], V[::-1]                     # np.interp needs increasing x

    ls = np.log(vs)
    inside = (ls >= Vi.min()) & (ls <= Vi.max())
    if int(inside.sum()) < min_points:
        raise ValueError(
            f"only {int(inside.sum())} of {xs.size} reference positions have an "
            f"InvOLS inside the dense pass's range "
            f"[{np.exp(Vi.min()):.3g}, {np.exp(Vi.max()):.3g}] m/V — the two "
            "passes barely overlap, so no scale can be fitted. Check that both "
            "passes really covered the same part of the beam.")

    xd_match = np.interp(ls[inside], Vi, Xi)
    A = np.polyfit(xd_match, xs[inside], 1)
    resid = xs[inside] - np.polyval(A, xd_match)
    ps = PositionScale(
        scale=A[0], offset=A[1], n_used=int(inside.sum()), resid_um=resid,
        excluded_x_um=xs[~inside],
        matched=list(zip(np.round(xd_match, 2), np.round(xs[inside], 2))),
        source="invols")
    if verbose:
        print(f"  position scale from InvOLS: {ps}")
        if abs(ps.scale - 1.0) > 0.02:
            print(f"    dense axis is compressed {100 * (1 - ps.scale):.1f}% — its "
                  f"span is really {ps.scale * (xd.max() - xd.min()):.1f} um, not "
                  f"{xd.max() - xd.min():.1f} um")
        if ps.excluded_x_um.size:
            print(f"    excluded {np.round(ps.excluded_x_um, 1)} um (InvOLS outside "
                  "the dense pass's range -> never actually covered by it)")
        if ps.rms_um is not None and ps.rms_um > 3.0:
            print(f"    WARNING: {ps.rms_um:.1f} um rms is large for an affine fit; "
                  "the error may not be a simple scale (check for drift).")
    return ps


def fit_position_scale_optical(images, x_um, spot_window_px=(690, 775),
                               search_px=(120, 700), verbose=True):
    """Independent cross-check on `fit_position_scale`, from the FAMap images.

    `_goto_and_prepare` saves an optical frame at every position. The laser spot
    is fixed in the frame (camera and detection optics move together) while the
    cantilever translates, so the spot-to-base-edge distance in pixels IS the
    spot's distance along the beam. Regressing it on the logged x gives
    px-per-commanded-µm; run this on both passes and the RATIO of the two slopes
    is the relative travel shortfall, with no pixel calibration needed.

    Used to validate the InvOLS method on Demo6Fullgrid: 0.8505 ± 0.0031 here
    (stable over six edge-detection variants) vs 0.8466 from InvOLS.

    Returns ``dict(slope_px_per_um, spot_px, base_px, dist_px, resid_px)``.
    Take the ratio of `slope_px_per_um` between two passes yourself, then build
    the correction with `PositionScale.from_scale(ratio, anchor)`.

    Needs Pillow. Returns None if it cannot read the images.
    """
    try:
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
    except Exception as e:                                  # pragma: no cover
        print(f"  optical scale check unavailable ({e})")
        return None
    x_um = np.asarray(x_um, float)
    spot, base = [], []
    for p in images:
        a = np.asarray(Image.open(p).convert("L"), dtype=float)
        r0 = int(np.argmax(a[:, 350:650].mean(1)))          # the bright beam row
        band = a[max(0, r0 - 25):r0 + 26, :].mean(0)
        s0, s1 = spot_window_px
        seg = band[s0:s1]
        w = np.clip(seg - np.percentile(band[s1 + 5:s1 + 130], 50), 0, None)
        spot.append(s0 + (np.arange(seg.size) * w).sum() / max(w.sum(), 1e-9))
        g0, g1 = search_px
        reg = band[g0:g1]
        i = int(np.argmax(np.gradient(reg)))                # steepest rise = base edge
        lo = np.percentile(reg[max(0, i - 60):max(1, i - 20)], 50)
        hi = np.percentile(reg[i + 20:i + 60], 50)
        lvl = 0.5 * (lo + hi)
        j = i
        while j > 0 and reg[j] > lvl:
            j -= 1
        base.append(g0 + j + (lvl - reg[j]) / (reg[j + 1] - reg[j]))
    spot = np.asarray(spot); base = np.asarray(base)
    dist = spot - base
    A = np.polyfit(x_um, dist, 1)
    resid = dist - np.polyval(A, x_um)
    if verbose:
        print(f"  optical: {A[0]:.4f} px per commanded um over {x_um.size} images, "
              f"residual rms {np.sqrt((resid ** 2).mean()):.1f} px")
    return dict(slope_px_per_um=float(A[0]), intercept_px=float(A[1]),
                spot_px=spot, base_px=base, dist_px=dist, resid_px=resid)


def peak_in_band(freq_Hz, Z, band_Hz=None):
    """Per-position peak |Z| in a band, and the frequency it occurred at.

    The f_res-shift-insensitive alternative to reading |Z| at one fixed
    frequency. On Demo6 the contact resonance wandered 386.9-387.9 kHz between
    positions while the linewidth was only ~1.3-2.8 kHz, so a fixed cut sampled
    a different part of every peak and turned position-to-position f_res jitter
    into fake amplitude structure (it accounted for a 1.46x -> 1.18x chunk of
    the apparent sparse/dense mismatch). Compare peaks, not cuts.
    """
    f = np.asarray(freq_Hz, float)
    Z = np.atleast_2d(np.asarray(Z, complex))
    m = np.ones(f.size, bool) if band_Hz is None else \
        (f >= float(band_Hz[0])) & (f <= float(band_Hz[1]))
    if m.sum() < 3:
        raise ValueError(f"band {band_Hz} keeps only {int(m.sum())} of {f.size} points")
    A = np.abs(Z[:, m])
    i = A.argmax(1)
    return A.max(1), f[m][i]


def compare_to_dense(mm, rec, x_dense, F_dense, Z_dense, dns_band_Hz=None,
                     position_scale=None, sparse_records=None,
                     dense_records=None, restrict_to_overlap=True,
                     amp_band_Hz=None):
    """Score the sparse reconstruction against a dense reference sweep.

    Returns a dict with the dense D-NS, the sparse D-NS, their difference, and
    the per-position relative map error. The dense sweep is interpolated onto
    `mm.freq` if the windows differ.

    Position-axis repair
    --------------------
    A dense sweep made of many short `DoLDMove` steps does not travel as far as
    its axis claims, so by default this refuses to compare the two passes on
    their logged axes alone. Supply ONE of:

    * `position_scale` — a `PositionScale`, or a float read as a pure scale, or
      `PositionScale.identity()` to assert deliberately that no correction is
      wanted;
    * `sparse_records` and `dense_records` — the two passes' record lists (or
      log DataFrames), and the scale is fitted from their InvOLS via
      `fit_position_scale`.

    Everything positional — the dense D-NS, the branch crossings, the
    reconstruction sampling — is then computed on the CORRECTED axis, because a
    D-NS read off a 15%-compressed axis is wrong by 15% of its distance from the
    anchor. The uncorrected values are returned alongside as `*_raw` so the size
    of the repair stays visible.

    With `restrict_to_overlap` (default), dense positions whose corrected value
    falls outside the sparse grid are dropped rather than silently extrapolated
    — on Demo6 the dense pass's real coverage stopped ~110 µm, so comparing it
    against sparse points at 85 µm was extrapolation dressed up as disagreement.

    Amplitude comparison uses `peak_in_band` over `amp_band_Hz` (defaults to the
    D-NS band), not a fixed-frequency cut.
    """
    import numpy as np
    from .lowrank import (resonance_index, dns_from_map, band_mask, dns_branch)

    x_dense = np.asarray(x_dense, float)
    Z = np.asarray(Z_dense, complex)
    if F_dense.shape != mm.freq.shape or not np.allclose(F_dense, mm.freq):
        Z = np.array([np.interp(mm.freq, F_dense, z.real)
                      + 1j * np.interp(mm.freq, F_dense, z.imag) for z in Z])
    band = dns_band_Hz or mm.dns_band_Hz
    bm = band_mask(mm.freq, band) if band else np.ones(mm.freq.size, bool)
    fb = mm.freq[bm]

    # -- resolve the position correction ---------------------------------- #
    if position_scale is None:
        if sparse_records is not None and dense_records is not None:
            position_scale = fit_position_scale(sparse_records, dense_records)
        else:
            raise ValueError(
                "compare_to_dense needs a position axis it can trust. Pass "
                "position_scale=..., or sparse_records=/dense_records= to fit "
                "it from InvOLS, or position_scale=PositionScale.identity() to "
                "state that the two passes are known to share an axis. A dense "
                "sweep built from many small DoLDMove steps generally does NOT "
                "share an axis with a sparse pass built from large ones.")
    if np.isscalar(position_scale):
        # A bare float means "the sweep under-travelled by this factor", and the
        # shortfall accumulates from wherever it started -- the largest x here,
        # since these sweeps park at the free end and step inward. Anchoring at
        # x = 0 instead would move the free end by tens of microns.
        anchor = float(np.max(x_dense))
        position_scale = PositionScale.from_scale(float(position_scale), anchor)
        print(f"  position_scale={position_scale.scale:.4f} read as a shortfall "
              f"accumulating from x={anchor:.1f} um -> {position_scale}")
    xc = position_scale.apply(x_dense)

    # -- dense estimators, on the corrected axis -------------------------- #
    ires_d = resonance_index(fb, Z[:, bm])
    dns_dense = dns_from_map(xc, Z[:, bm], fb, ires_d)
    cross_dense, _ = dns_branch(xc, Z[:, bm], fb, ires_d)
    dns_dense_raw = dns_from_map(x_dense, Z[:, bm], fb, ires_d)

    # -- overlap with the sparse grid -------------------------------------- #
    lo, hi = float(mm.x_grid.min()), float(mm.x_grid.max())
    keep = np.ones(xc.size, bool) if not restrict_to_overlap else \
        (xc >= lo - 1e-9) & (xc <= hi + 1e-9)
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"  dropped {n_drop} of {xc.size} dense positions: corrected "
              f"positions outside the sparse grid [{lo:.1f}, {hi:.1f}] um. "
              "Comparing there would be extrapolation, not disagreement.")
    if not keep.any():
        raise ValueError(
            "after correction NO dense position lies inside the sparse grid — "
            "the two passes did not cover the same part of the beam.")

    idx = [int(np.argmin(np.abs(mm.x_grid - x))) for x in xc[keep]]
    Zr = rec["Zrec"][idx]
    Zk = Z[keep]
    rel = np.abs(Zr - Zk).mean(1) / (np.abs(Zk).mean(1) + 1e-30)

    # -- amplitude, peak-in-band rather than a fixed cut ------------------- #
    ab = amp_band_Hz or band
    pk_d, fpk_d = peak_in_band(mm.freq, Zk, ab)
    pk_r, fpk_r = peak_in_band(mm.freq, Zr, ab)
    ratio = pk_r / np.where(pk_d == 0, np.nan, pk_d)

    return dict(dns_dense=dns_dense, crossings_dense=cross_dense,
                dns_sparse=rec["dns"], delta=rec["dns"] - dns_dense,
                dns_dense_raw=dns_dense_raw,
                delta_raw=rec["dns"] - dns_dense_raw,
                position_scale=position_scale,
                x_dense=x_dense[keep], x_dense_corrected=xc[keep],
                x_dense_all=x_dense, x_dense_corrected_all=xc,
                kept=keep, n_dropped=n_drop,
                rel_err=rel, rel_err_mean=float(rel.mean()),
                rel_err_max=float(rel.max()),
                peak_dense=pk_d, peak_sparse=pk_r,
                peak_freq_dense=fpk_d, peak_freq_sparse=fpk_r,
                peak_ratio=ratio,
                peak_ratio_median=float(np.nanmedian(ratio)),
                amp_band_Hz=ab,
                Z_dense_on_grid=Zk, ires_dense=int(np.flatnonzero(bm)[ires_d]),
                n_sparse=mm.n, n_dense=int(keep.sum()))

