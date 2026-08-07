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


def d_optimal_order(x_um: np.ndarray, rank: int, seeds=None, n_select=None):
    """Greedy D-optimal ordering of positions for the rank-`rank` spatial basis.

    Adds positions that maximize det(B_sel^T B_sel) — i.e. that best condition
    the spatial fit and minimize reconstruction variance. Returns a list of
    indices into `x_um`, length `n_select` (default: all)."""
    x_um = np.asarray(x_um, float)
    n = x_um.size
    B = chebyshev_basis(x_um, rank)
    if seeds is None:
        seeds = [int(np.argmin(x_um)), int(np.argmax(x_um))]
    sel = list(dict.fromkeys(seeds))
    target = n if n_select is None else min(n_select, n)
    while len(sel) < target:
        best, best_d = None, -np.inf
        for p in range(n):
            if p in sel:
                continue
            M = B[sel + [p]]
            d = np.linalg.slogdet(M.T @ M)[1]
            if d > best_d:
                best_d, best = d, p
        sel.append(best)
    return sel


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
    resonance frequency fres. Robust when a genuine branch crossing exists."""
    fres = freq[ires]
    W = window_frac * (freq.max() - freq.min())
    lo = max(int(np.searchsorted(freq, fres - W)), 0)
    hi = min(int(np.searchsorted(freq, fres + W)), freq.size)
    fa = np.array([freq[lo + int(np.argmin(np.abs(Zrec[i, lo:hi])))]
                   for i in range(Zrec.shape[0])])
    g = fa - fres
    idx = np.where(np.diff(np.sign(g)) != 0)[0]
    if idx.size == 0:
        return np.nan
    ref = guess_um if guess_um is not None else x[x.size // 2]
    xc = x[idx] + (x[idx + 1] - x[idx]) * g[idx] / (g[idx] - g[idx + 1])
    return float(xc[int(np.argmin(np.abs(xc - ref)))])


def _dns_onres_crossing(x, Zrec, freq, ires, guess_um):
    """Signed zero crossing of the phase-rotated on-resonance response along
    position, anchored to the on-resonance amplitude minimum."""
    zres = Zrec[:, ires]
    amp = np.abs(zres)
    guess = guess_um if guess_um is not None else x[int(np.argmin(amp))]
    istrong = int(np.argmax(amp))
    v = (zres * np.exp(-1j * np.angle(zres[istrong]))).real
    return _signed_zero_crossing(x, v, x_guess=guess)


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

    Positions are chosen on a fixed candidate grid `x_grid_um`; every spectrum
    is stored on a common frequency grid `freq_grid`. Typical loop:

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
                 min_positions=None, dns_ci_tol_um=1.0, rng=None):
        self.x_grid = np.asarray(x_grid_um, float)
        # freq_grid may be None; it is then adopted from the first measurement,
        # so the common grid matches the instrument's actual tune window.
        self.freq = None if freq_grid is None else np.asarray(freq_grid, float)
        self.rank = int(rank)
        self.min_positions = min_positions or (self.rank + 1)
        self.dns_ci_tol_um = dns_ci_tol_um
        self.rng = rng or np.random.default_rng(0)
        # D-optimal ordering of the whole grid at the target rank
        seed_idx = None
        if seeds_um is not None:
            seed_idx = [int(np.argmin(np.abs(self.x_grid - s))) for s in seeds_um]
        self._order = d_optimal_order(self.x_grid, self.rank, seeds=seed_idx)
        self._measured = {}     # grid index -> complex spectrum on self.freq
        self._last = None

    # -- data ------------------------------------------------------------- #
    @property
    def n(self):
        return len(self._measured)

    def add_measurement(self, x_um, freq, Z):
        """Store a measured spectrum, interpolated onto the common freq grid,
        at the nearest grid position."""
        i = int(np.argmin(np.abs(self.x_grid - x_um)))
        Z = np.asarray(Z, complex)
        freq = np.asarray(freq, float)
        if self.freq is None:                      # adopt instrument's tune grid
            self.freq = freq.copy()
        if freq.shape != self.freq.shape or not np.allclose(freq, self.freq):
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
        if self.n < 2:
            raise RuntimeError("need at least 2 positions")
        sel = sorted(self._measured)
        Zsel = np.array([self._measured[i] for i in sel])
        rank = min(self.rank, len(sel))
        rec = reconstruct_map(self.x_grid, sel, Zsel, rank)
        ires = resonance_index(self.freq, Zsel)
        dns = dns_from_map(self.x_grid, rec["Zrec"], self.freq, ires)
        # bootstrap D-NS over measurement noise (median residual as noise scale)
        noise = float(np.median(rec["sig"]))
        B, Bs = rec["B"], rec["B"][sel]
        dsamp = []
        for _ in range(nboot):
            zb = Zsel + noise * (self.rng.standard_normal(Zsel.shape)
                                 + 1j * self.rng.standard_normal(Zsel.shape)) / np.sqrt(2)
            cb = np.linalg.lstsq(Bs, zb, rcond=None)[0]
            d = dns_from_map(self.x_grid, B @ cb, self.freq, ires, guess_um=dns)
            if np.isfinite(d):
                dsamp.append(d)
        dsamp = np.array(dsamp)
        ci = (np.percentile(dsamp, 97.5) - np.percentile(dsamp, 2.5)
              if dsamp.size else np.nan)
        self._last = dict(sel_idx=sel, x_sel=self.x_grid[sel], ires=ires,
                          dns=dns, dns_ci=ci, dns_samples=dsamp, **rec)
        return self._last

    def converged(self):
        if self._last is None:
            return False
        return (self.n >= self.min_positions
                and np.isfinite(self._last["dns_ci"])
                and self._last["dns_ci"] < self.dns_ci_tol_um)
