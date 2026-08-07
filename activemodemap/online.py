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


def compare_to_dense(mm, rec, x_dense, F_dense, Z_dense, dns_band_Hz=None):
    """Score the sparse reconstruction against a dense reference sweep.

    Returns a dict with the dense D-NS, the sparse D-NS, their difference, and
    the per-position relative map error. The dense sweep is interpolated onto
    `mm.freq` if the windows differ.
    """
    import numpy as np
    from .lowrank import (resonance_index, dns_from_map, band_mask, dns_branch)

    Z = np.asarray(Z_dense, complex)
    if F_dense.shape != mm.freq.shape or not np.allclose(F_dense, mm.freq):
        Z = np.array([np.interp(mm.freq, F_dense, z.real)
                      + 1j * np.interp(mm.freq, F_dense, z.imag) for z in Z])
    band = dns_band_Hz or mm.dns_band_Hz
    bm = band_mask(mm.freq, band) if band else np.ones(mm.freq.size, bool)
    fb = mm.freq[bm]

    ires_d = resonance_index(fb, Z[:, bm])
    dns_dense = dns_from_map(x_dense, Z[:, bm], fb, ires_d)
    cross_dense, _ = dns_branch(x_dense, Z[:, bm], fb, ires_d)

    # sparse reconstruction sampled at the dense positions
    idx = [int(np.argmin(np.abs(mm.x_grid - x))) for x in x_dense]
    Zr = rec["Zrec"][idx]
    rel = np.abs(Zr - Z).mean(1) / (np.abs(Z).mean(1) + 1e-30)

    return dict(dns_dense=dns_dense, crossings_dense=cross_dense,
                dns_sparse=rec["dns"], delta=rec["dns"] - dns_dense,
                rel_err=rel, rel_err_mean=float(rel.mean()),
                rel_err_max=float(rel.max()), x_dense=x_dense,
                Z_dense_on_grid=Z, ires_dense=int(np.flatnonzero(bm)[ires_d]),
                n_sparse=mm.n, n_dense=len(x_dense))
