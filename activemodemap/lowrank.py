"""Model-light low-rank mode-shape reconstruction.

The position-frequency response is separable and low rank,

    z(x, f) = sum_n  phi_n(x) a_n(f),

so it can be reconstructed from a few detection positions without assuming
anything about the shape of the frequency response. We represent the spatial
dependence with a small smooth basis (Chebyshev polynomials over the accessible
span) and, independently at each frequency, fit the complex basis coefficients
to the spectra measured at the selected positions by linear least squares.

Nothing is smoothed along the frequency axis, so sharp resonance and
antiresonance features are preserved exactly; the only assumption is that the
mode shape is smooth in x. This is the reconstruction used on real-probe data,
where beam theory is not exact.

Two entry points:
  * `reconstruct_map(...)`  — one-shot reconstruction from a fixed position set
  * `LowRankModeMap`        — incremental/online manager driving an AL loop
                              (D-optimal position selection, uncertainty, D-NS)
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial import chebyshev as _cheb


# --------------------------------------------------------------------------- #
#  Spatial basis                                                              #
# --------------------------------------------------------------------------- #
def chebyshev_basis(x_um: np.ndarray, rank: int) -> np.ndarray:
    """Chebyshev polynomials (degree 0..rank-1) evaluated on the position grid,
    with x mapped to [-1, 1] over its own range. Returns (n_positions, rank)."""
    x = np.asarray(x_um, float)
    span = x.max() - x.min()
    xn = 2 * (x - x.min()) / span - 1 if span > 0 else np.zeros_like(x)
    B = np.zeros((x.size, rank))
    for k in range(rank):
        c = np.zeros(rank); c[k] = 1.0
        B[:, k] = _cheb.chebval(xn, c)
    return B


def d_optimal_order(x_um: np.ndarray, rank: int, seeds=None, n_select=None,
                    ridge: float = 1e-9):
    """Greedy D-optimal ordering of positions for the rank-`rank` spatial basis.

    Adds positions that maximize det(B_sel^T B_sel) — i.e. that best condition
    the spatial fit and minimize reconstruction variance. Returns a list of
    indices into `x_um`, length `n_select` (default: all).

    While fewer than `rank` positions have been chosen, B_sel^T B_sel is
    rank-deficient and its determinant is zero for *every* candidate, so a
    plain det criterion is decided by floating-point noise and happily picks
    adjacent, nearly-redundant positions. We therefore score each candidate at
    the highest basis order it can actually support (`min(rank, len(sel)+1)`)
    and add a tiny ridge, which makes the early picks well posed and recovers
    the expected Chebyshev-like spread.
    """
    x_um = np.asarray(x_um, float)
    n = x_um.size
    if seeds is None:
        seeds = [int(np.argmin(x_um)), int(np.argmax(x_um))]
    sel = list(dict.fromkeys(int(s) for s in seeds))
    target = n if n_select is None else min(n_select, n)
    while len(sel) < target:
        # only ask for as many basis functions as the current set can support
        r = int(min(rank, len(sel) + 1))
        B = chebyshev_basis(x_um, r)
        scale = float(np.trace(B.T @ B)) / max(r, 1) or 1.0
        best, best_d = None, -np.inf
        for p in range(n):
            if p in sel:
                continue
            M = B[sel + [p]]
            A = M.T @ M + ridge * scale * np.eye(r)
            d = np.linalg.slogdet(A)[1]
            if d > best_d:
                best_d, best = d, p
        if best is None:                      # nothing left to add
            break
        sel.append(best)
    return sel


def band_mask(freq_Hz, band_Hz):
    """Boolean mask selecting ``band_Hz=(lo, hi)`` out of a frequency axis.

    Restricting the reconstruction to one eigenmode's band is not cosmetic:
    `dns_from_map` searches for the antiresonance notch within
    ``window_frac * (freq.max() - freq.min())`` of the resonance, so a window
    far wider than the mode makes that search window meaningless.
    """
    f = np.asarray(freq_Hz, float)
    return (f >= float(band_Hz[0])) & (f <= float(band_Hz[1]))


# --------------------------------------------------------------------------- #
#  Core reconstruction                                                        #
# --------------------------------------------------------------------------- #
def reconstruct_map(x_grid_um, sel_idx, Z_sel, rank):
    """Reconstruct the full complex map from spectra at selected positions.

    x_grid_um : (npos,) full candidate position grid (µm)
    sel_idx   : indices into x_grid_um that were measured
    Z_sel     : (len(sel_idx), nfreq) complex spectra at those positions
    rank      : spatial basis dimension (must be <= len(sel_idx))

    Returns dict with Zrec (npos, nfreq) complex, std (npos, nfreq) amplitude
    1-sigma, coef, and the fixed spatial basis B.
    """
    x_grid_um = np.asarray(x_grid_um, float)
    Z_sel = np.asarray(Z_sel, complex)
    rank = min(rank, len(sel_idx))
    B = chebyshev_basis(x_grid_um, rank)
    Bs = B[sel_idx]
    coef = np.linalg.lstsq(Bs, Z_sel, rcond=None)[0]          # (rank, nfreq)
    Zrec = B @ coef
    # per-frequency residual std at the measured points (noise + rank truncation)
    resid = Bs @ coef - Z_sel
    dof = max(len(sel_idx) - rank, 1)
    sig = np.sqrt((np.abs(resid) ** 2).sum(0) / dof)          # (nfreq,)
    # spatial leverage -> predictive variance shape across positions
    G = np.linalg.inv(Bs.conj().T @ Bs)
    lev = np.einsum("ij,jk,ik->i", B, G, B.conj()).real       # (npos,)
    std = np.sqrt(np.outer(lev, sig ** 2))                    # (npos, nfreq)
    return dict(Zrec=Zrec, std=std, coef=coef, B=B, sig=sig, lev=lev)


# --------------------------------------------------------------------------- #
#  Operating-point extraction                                                 #
# --------------------------------------------------------------------------- #
def _signed_zero_crossing(x, v, x_guess=None):
    s = np.sign(v)
    idx = np.where(np.diff(s) != 0)[0]
    if idx.size == 0:
        return float(x[int(np.argmin(np.abs(v)))])
    xc = x[idx] + (x[idx + 1] - x[idx]) * v[idx] / (v[idx] - v[idx + 1])
    if x_guess is None:
        return float(xc[-1])                      # nearest the free end
    return float(xc[int(np.argmin(np.abs(xc - x_guess)))])


def resonance_index(freq, Z_sel):
    """Frequency index of the contact resonance (peak of the strongest column)."""
    strong = int(np.argmax(np.abs(Z_sel).max(axis=1)))
    return int(np.argmax(np.abs(Z_sel[strong])))


def _dns_branch_crossing(x, Zrec, freq, ires, guess_um, window_frac):
    """Position where the antiresonance-notch frequency fa(x) crosses the
    resonance frequency fres. Robust when a genuine branch crossing exists.

    Positions whose amplitude minimum falls on the EDGE of the search window
    have no notch inside the window at all; their fa is meaningless and is
    marked NaN rather than being pinned to the boundary. Without this, fa(x)
    jumps between the two window edges wherever the notch has left the window
    and those jumps are picked up as spurious "crossings" — which is how a
    higher mode can report a null spot in the middle of the beam where the map
    shows nothing.
    """
    fres = freq[ires]
    W = window_frac * (freq.max() - freq.min())
    lo = max(int(np.searchsorted(freq, fres - W)), 0)
    hi = min(int(np.searchsorted(freq, fres + W)), freq.size)
    if hi - lo < 5:
        return np.nan
    fa = np.full(Zrec.shape[0], np.nan)
    for i in range(Zrec.shape[0]):
        j = int(np.argmin(np.abs(Zrec[i, lo:hi])))
        if j == 0 or j == hi - lo - 1:          # notch outside the window
            continue
        fa[i] = freq[lo + j]
    g = fa - fres
    ok = np.isfinite(g)
    # sign changes between ADJACENT positions that both have a real notch
    pair = ok[:-1] & ok[1:]
    sgn = np.zeros(g.size - 1, bool)
    sgn[pair] = np.sign(g[:-1][pair]) != np.sign(g[1:][pair])
    idx = np.where(sgn)[0]
    if idx.size == 0:
        return np.nan
    xc = x[idx] + (x[idx + 1] - x[idx]) * g[idx] / (g[idx] - g[idx + 1])
    # Default to the crossing nearest the FREE END (largest x), matching both
    # the physics -- the first contact mode's displacement null sits a few um
    # in from the tip -- and `_signed_zero_crossing`, which has always resolved
    # ties that way. Defaulting to the middle of the grid instead made this
    # estimator pick a base-side branch whenever the position grid extended
    # beyond the measured span.
    ref = guess_um if guess_um is not None else x[-1]
    return float(xc[int(np.argmin(np.abs(xc - ref)))])


def dns_branch(x, Zrec, freq, ires, window_frac=0.35, merge_tol_um=None):
    """All antiresonance-branch crossings of the resonance, and fa(x) itself.

    Returns ``(crossings_um, fa_Hz)`` with NaN in `fa` wherever the notch lies
    outside the search window. Use this to inspect a map before trusting a
    single D-NS number — a mode with more than one null, or a window that is
    too narrow, shows up immediately here.
    """
    fres = freq[ires]
    W = window_frac * (freq.max() - freq.min())
    lo = max(int(np.searchsorted(freq, fres - W)), 0)
    hi = min(int(np.searchsorted(freq, fres + W)), freq.size)
    fa = np.full(Zrec.shape[0], np.nan)
    for i in range(Zrec.shape[0]):
        j = int(np.argmin(np.abs(Zrec[i, lo:hi])))
        if 0 < j < hi - lo - 1:
            fa[i] = freq[lo + j]
    g = fa - fres
    ok = np.isfinite(g)
    pair = ok[:-1] & ok[1:]
    sgn = np.zeros(max(g.size - 1, 0), bool)
    sgn[pair] = np.sign(g[:-1][pair]) != np.sign(g[1:][pair])
    idx = np.where(sgn)[0]
    xc = (np.asarray(x)[idx] + (np.asarray(x)[idx + 1] - np.asarray(x)[idx])
          * g[idx] / (g[idx] - g[idx + 1])) if idx.size else np.array([])
    if xc.size > 1:
        xa = np.asarray(x, float)
        tol = (merge_tol_um if merge_tol_um is not None
               else float(np.median(np.diff(xa))) if xa.size > 1 else 0.0)
        xc = np.sort(xc)
        groups, cur = [], [xc[0]]
        for v in xc[1:]:
            if v - cur[-1] <= tol:
                cur.append(v)
            else:
                groups.append(cur); cur = [v]
        groups.append(cur)
        xc = np.array([np.mean(gp) for gp in groups])
    return xc, fa


def _dns_onres_crossing(x, Zrec, freq, ires, guess_um):
    """Signed zero crossing of the phase-rotated on-resonance response along
    position, anchored to the on-resonance amplitude minimum."""
    zres = Zrec[:, ires]
    amp = np.abs(zres)
    guess = guess_um if guess_um is not None else x[int(np.argmin(amp))]
    istrong = int(np.argmax(amp))
    v = (zres * np.exp(-1j * np.angle(zres[istrong]))).real
    return _signed_zero_crossing(x, v, x_guess=guess)


def spatial_null(x_grid_um, Zrec, freq, i_freq=None, guess_um=None):
    """Position where the response vanishes along x, at one frequency.

    A signed zero crossing of the phase-rotated response, so it finds a genuine
    sign change rather than merely a shallow amplitude minimum.

    This -- not the antiresonance branch crossing -- is the right estimator for
    the ELECTROSTATIC blind spot. The branch crossing locates where the
    antiresonance notch frequency meets the resonance, which is a property of the
    coupled dynamics and comes out at essentially the same position for both
    channels, so it cannot distinguish D-NS from D-ESBS. The electrostatic blind
    spot is a spatial null of the electrostatic drive's detected response, which
    is what this measures.

    `i_freq` defaults to the strongest frequency in the map.
    """
    x = np.asarray(x_grid_um, float)
    if i_freq is None:
        i_freq = int(np.argmax(np.abs(Zrec).max(axis=0)))
    return _dns_onres_crossing(x, Zrec, freq, int(i_freq), guess_um)


def dns_from_map(x_grid_um, Zrec, freq, ires, guess_um=None, window_frac=0.35):
    """Displacement null spot (position where the resonant response vanishes).

    Primary estimator: the detection position where the antiresonance branch
    crosses the resonance branch — i.e. where the antiresonance-notch frequency
    fa(x) equals the resonance frequency fres. This is stable because the notch
    is a deep, well-localized feature, unlike the tiny on-resonance amplitude
    right at the null. If no branch crossing exists in the search window (some
    probes / bands), fall back to the signed zero crossing of the on-resonance
    response, anchored to its amplitude minimum. `window_frac` sets the notch
    search half-window as a fraction of the frequency span."""
    x = np.asarray(x_grid_um, float)
    val = _dns_branch_crossing(x, Zrec, freq, ires, guess_um, window_frac)
    if not np.isfinite(val):
        val = _dns_onres_crossing(x, Zrec, freq, ires, guess_um)
    return val


# --------------------------------------------------------------------------- #
#  Online / incremental manager                                              #
# --------------------------------------------------------------------------- #
class LowRankModeMap:
    """Incremental low-rank reconstruction driving an active-learning loop.

    Positions are chosen on a fixed candidate grid `x_grid_um`, in the library
    frame: **x = 0 is the clamped base and x increases toward the free end**
    (`_signed_zero_crossing` resolves ties to the largest x as "nearest the free
    end", so this must not be flipped). Every spectrum is stored on a common
    frequency grid `freq_grid`.

    `start_near_um` sets which seed is visited first -- pass the coordinate the
    detection spot is parked at, e.g. `x_grid[-1]` when starting at the free
    end. Typical loop:

        mm = LowRankModeMap(x_grid_um, freq_grid, rank=5)
        for _ in range(max_positions):
            x = mm.next_position()
            freq, Z = instrument.measure_at(x)      # your hardware call
            mm.add_measurement(x, freq, Z)
            if mm.n >= mm.min_positions:
                rec = mm.reconstruct()
                if mm.converged(): break
    """

    def __init__(self, x_grid_um, freq_grid=None, rank=5, seeds_um=None,
                 min_positions=None, dns_ci_tol_um=1.0, rng=None,
                 start_near_um=None, dns_band_Hz=None, stable_over=3):
        self.x_grid = np.asarray(x_grid_um, float)
        # freq_grid may be None; it is then adopted from the first measurement,
        # so the common grid matches the instrument's actual tune window.
        self.freq = None if freq_grid is None else np.asarray(freq_grid, float)
        self.rank = int(rank)
        # rank+2 gives 2 residual degrees of freedom. At npos == rank the fit is
        # exactly determined: the residual is identically zero, so sig = 0, the
        # bootstrap CI is 0, and converged() fires immediately on a fit that has
        # not been tested against anything. rank+1 leaves 1 dof, which is barely
        # better. rank+2 is the smallest honest default.
        self.min_positions = min_positions or (self.rank + 2)
        if self.min_positions < self.rank:
            raise ValueError(
                f"min_positions ({self.min_positions}) < rank ({self.rank}): the "
                "spatial fit would be underdetermined. Lower `rank` instead -- "
                "that is the knob that reduces how many positions you need.")
        if self.min_positions < self.rank + 2:
            print(f"  warning: min_positions={self.min_positions} with rank="
                  f"{self.rank} leaves only {max(self.min_positions - self.rank, 0)} "
                  "residual dof, so the bootstrap CI will understate the true "
                  f"uncertainty. {self.rank + 2} or more is safer.")
        self.dns_ci_tol_um = dns_ci_tol_um
        # Number of consecutive reconstructions whose D-NS must agree to within
        # dns_ci_tol_um before we call it converged. The bootstrap CI alone is
        # over-optimistic -- it draws its noise scale from the fit residual, and
        # at min_positions = rank + 1 there are only a couple of residual dof --
        # so on its own it trips at the first opportunity regardless of data
        # quality. Requiring the estimate to actually stop moving is the honest
        # test. Set to 1 to recover the old CI-only behaviour.
        self.stable_over = max(int(stable_over), 1)
        self.rng = rng or np.random.default_rng(0)
        # D-optimal ordering of the whole grid at the target rank
        seed_idx = None
        if seeds_um is not None:
            seed_idx = [int(np.argmin(np.abs(self.x_grid - s))) for s in seeds_um]
        if start_near_um is not None and seed_idx:
            # Visit the seed the detection spot is already parked at first, then
            # work away from it. The seed SET is unchanged, so D-optimality is
            # untouched -- this only reorders the visits, which saves a full
            # traverse of the beam before the first measurement (and matters if
            # the run is aborted part-way).
            seed_idx = sorted(seed_idx,
                              key=lambda i: abs(self.x_grid[i] - float(start_near_um)))
        self.start_near_um = start_near_um
        self._order = d_optimal_order(self.x_grid, self.rank, seeds=seed_idx)
        # Band used ONLY for locating the resonance and the null spot. The map
        # itself is always reconstructed over the FULL frequency axis: the fit is
        # an independent least-squares solve at each frequency, so covering the
        # whole tune costs nothing and no information is thrown away. Only the
        # D-NS estimator needs a band, because its antiresonance search window is
        # a fraction of the span it is handed.
        self.dns_band_Hz = None if dns_band_Hz is None else (
            float(min(dns_band_Hz)), float(max(dns_band_Hz)))
        self._measured = {}     # grid index -> complex spectrum on self.freq
        self._last = None
        self.history = []       # [{n, dns, dns_ci}] one entry per reconstruct()

    # -- data ------------------------------------------------------------- #
    @property
    def n(self):
        return len(self._measured)

    #: refuse to build a reconstruction on fewer frequency points than this —
    #: a short spectrum means a truncated instrument read, not a real window
    min_freq_points = 32

    def add_measurement(self, x_um, freq, Z):
        """Store a measured spectrum, interpolated onto the common freq grid,
        at the nearest grid position."""
        i = int(np.argmin(np.abs(self.x_grid - x_um)))
        Z = np.asarray(Z, complex)
        freq = np.asarray(freq, float)
        if freq.size < self.min_freq_points:
            raise ValueError(
                f"spectrum has only {freq.size} frequency points "
                f"({freq.min():.0f}-{freq.max():.0f} Hz); expected the full tune "
                f"window (>= {self.min_freq_points}). A short spectrum here means "
                "the instrument read was truncated — fix that rather than "
                "reconstructing from it.")
        # Non-finite samples must never reach lstsq: they propagate into every
        # reconstructed column and NaN-poison downstream reductions (max, std).
        good = np.isfinite(freq) & np.isfinite(Z.real) & np.isfinite(Z.imag)
        if not good.all():
            if good.sum() < self.min_freq_points:
                raise ValueError(
                    f"spectrum at x={x_um:.1f} um has only {int(good.sum())} finite "
                    f"points out of {freq.size}. Check the tune in Igor -- the "
                    "Amp/Phase waves are probably NaN outside the swept range.")
            print(f"    note: dropping {int((~good).sum())} non-finite points from "
                  f"the spectrum at x={x_um:.1f} um")
            freq, Z = freq[good], Z[good]
        if self.freq is None:                      # adopt instrument's tune grid
            self.freq = freq.copy()
        if freq.shape != self.freq.shape or not np.allclose(freq, self.freq):
            # Report it: a point or two of difference is non-finite trimming, but a
            # shifted window means the tune is recentring and the map is not
            # comparable position to position. Silently interpolating hid that.
            drift = max(abs(freq[0] - self.freq[0]), abs(freq[-1] - self.freq[-1]))
            span = float(self.freq[-1] - self.freq[0]) or 1.0
            msg = (f"    resampling x={x_um:.1f} um onto the common grid "
                   f"({freq.size} -> {self.freq.size} pts, window edges move "
                   f"{drift / 1e3:.2f} kHz)")
            if drift / span > 0.02:
                msg += "  ** window shifted >2% of the span: is resonance "\
                       "tracking ON in Igor? **"
            print(msg)
            Zr = np.interp(self.freq, freq, Z.real)
            Zi = np.interp(self.freq, freq, Z.imag)
            Z = Zr + 1j * Zi
        self._measured[i] = Z
        return i

    # -- acquisition ------------------------------------------------------ #
    def next_position(self):
        """Next detection position (µm) from the D-optimal order not yet
        measured. Once the D-optimal set is exhausted, refine near the current
        D-NS estimate."""
        for i in self._order:
            if i not in self._measured:
                return float(self.x_grid[i])
        # exhausted -> refine around the current null estimate
        if self._last is not None and np.isfinite(self._last["dns"]):
            cand = [i for i in range(self.x_grid.size) if i not in self._measured]
            if cand:
                j = min(cand, key=lambda i: abs(self.x_grid[i] - self._last["dns"]))
                return float(self.x_grid[j])
        return float(self.x_grid[self.rng.integers(self.x_grid.size)])

    # -- reconstruction --------------------------------------------------- #
    def reconstruct(self, nboot=200):
        """Reconstruct the map over the WHOLE frequency axis.

        `dns_band_Hz` restricts only where the resonance and the null spot are
        located, not what is reconstructed -- `rec['Zrec']` always spans the full
        tune. `rec['ires']` indexes the full axis (so it can be plotted directly)
        and `rec['band_mask']` marks the sub-band the D-NS came from.
        """
        if self.n < 2:
            raise RuntimeError("need at least 2 positions")
        sel = sorted(self._measured)
        Zsel = np.array([self._measured[i] for i in sel])
        rank = min(self.rank, len(sel))

        # full-spectrum reconstruction
        rec = reconstruct_map(self.x_grid, sel, Zsel, rank)

        # locate the resonance / null inside the D-NS band only
        if self.dns_band_Hz is not None:
            bm = band_mask(self.freq, self.dns_band_Hz)
            if bm.sum() < 8:
                raise ValueError(
                    f"dns_band_Hz={self.dns_band_Hz} keeps only {int(bm.sum())} of "
                    f"{self.freq.size} points (tune window "
                    f"{self.freq[0]:.0f}-{self.freq[-1]:.0f} Hz)")
        else:
            bm = np.ones(self.freq.size, bool)
        fb = self.freq[bm]
        Zb = Zsel[:, bm]
        Zrec_b = rec["Zrec"][:, bm]
        ires_b = resonance_index(fb, Zb)
        ires = int(np.flatnonzero(bm)[ires_b])          # index on the full axis
        dns = dns_from_map(self.x_grid, Zrec_b, fb, ires_b)
        crossings, fa = dns_branch(self.x_grid, Zrec_b, fb, ires_b)

        # bootstrap the D-NS over measurement noise, in the band
        noise = float(np.median(rec["sig"][bm]))
        B, Bs = rec["B"], rec["B"][sel]
        dsamp = []
        for _ in range(nboot):
            zb = Zb + noise * (self.rng.standard_normal(Zb.shape)
                               + 1j * self.rng.standard_normal(Zb.shape)) / np.sqrt(2)
            cb = np.linalg.lstsq(Bs, zb, rcond=None)[0]
            d = dns_from_map(self.x_grid, B @ cb, fb, ires_b, guess_um=dns)
            if np.isfinite(d):
                dsamp.append(d)
        dsamp = np.array(dsamp)
        ci = (np.percentile(dsamp, 97.5) - np.percentile(dsamp, 2.5)
              if dsamp.size else np.nan)

        self._last = dict(sel_idx=sel, x_sel=self.x_grid[sel], ires=ires,
                          ires_band=ires_b, band_mask=bm,
                          dns_band_Hz=self.dns_band_Hz,
                          dns=dns, dns_ci=ci, dns_samples=dsamp,
                          crossings=crossings, fa=fa,
                          Zsel=Zsel, **rec)
        self.history.append(dict(n=self.n, dns=dns, dns_ci=ci))
        return self._last

    def rank_sensitivity(self, ranks=None, verbose=True):
        """Re-fit the same measurements at several ranks and report the spread.

        The bootstrap CI only propagates measurement noise through a fit whose
        rank is assumed correct, so it is blind to the dominant error here:
        choosing the basis size. Re-fitting at neighbouring ranks exposes that
        directly, and the spread across them is a much more honest uncertainty
        than the CI. Costs nothing -- no new measurements.

        Defaults to `rank-1, rank, rank+1`. Scanning much wider is misleading
        rather than conservative: rank 2-3 is too stiff to represent the null
        (systematically ~3 um out in simulation) and rank >= n-1 approaches an
        exactly determined fit, so including them inflates the spread with
        configurations you would never use. Pass `ranks=` explicitly to look
        wider -- worth doing once per probe to confirm rank 4-5 is the plateau.

        Returns ``{rank: dns}`` plus 'spread' and 'median'.
        """
        if self.n < 3:
            raise RuntimeError("need at least 3 positions")
        if ranks is None:
            ranks = [r for r in (self.rank - 1, self.rank, self.rank + 1)
                     if 2 <= r <= self.n]
        sel = sorted(self._measured)
        Zsel = np.array([self._measured[i] for i in sel])
        if self.dns_band_Hz is not None:
            bm = band_mask(self.freq, self.dns_band_Hz)
        else:
            bm = np.ones(self.freq.size, bool)
        fb, Zb = self.freq[bm], Zsel[:, bm]
        ires_b = resonance_index(fb, Zb)
        out = {}
        for r in ranks:
            rr = min(r, len(sel))
            rec = reconstruct_map(self.x_grid, sel, Zb, rr)
            out[r] = dns_from_map(self.x_grid, rec["Zrec"], fb, ires_b)
        vals = np.array([v for v in out.values() if np.isfinite(v)])
        # drop the exactly-determined fit: zero residual, not a real estimate
        usable = np.array([v for r, v in out.items()
                           if np.isfinite(v) and r < len(sel)])
        res = dict(out)
        res["median"] = float(np.median(usable)) if usable.size else np.nan
        res["spread"] = float(usable.max() - usable.min()) if usable.size > 1 else np.nan
        if verbose:
            print(f"  D-NS vs rank ({self.n} positions):")
            for r in ranks:
                flag = "  <- exactly determined, ignore" if r >= len(sel) else ""
                print(f"    rank {r}: {out[r]:8.2f} um{flag}")
            print(f"  median {res['median']:.2f} um, spread across neighbouring "
                  f"ranks {res['spread']:.2f} um")
            print("  ^ use that spread, not the bootstrap CI, as the uncertainty "
                  "you quote")
        return res

    def converged(self):
        """True when the D-NS has both a tight CI and has stopped moving."""
        if self._last is None or self.n < self.min_positions:
            return False
        if not np.isfinite(self._last["dns_ci"]):
            return False
        if self._last["dns_ci"] >= self.dns_ci_tol_um:
            return False
        if self.stable_over <= 1:
            return True
        recent = [e["dns"] for e in self.history[-self.stable_over:]]
        if len(recent) < self.stable_over:
            return False
        recent = np.asarray(recent, float)
        return bool(np.all(np.isfinite(recent))
                    and (recent.max() - recent.min()) < self.dns_ci_tol_um)
