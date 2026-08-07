#!/usr/bin/env python
"""Rebuild the low-rank mode-shape reconstruction from saved Igor tune files.

The tune ``.txt`` files Igor writes always contain the complete spectrum, so a
run whose live reconstruction was wrong can be recovered offline without going
back on the microscope. Use this to re-analyse any folder of
``Tune_<base>_X<nm>_DC<bias>_L<load>nN_<idx>.txt`` files, and to choose the
analysis band per eigenmode.

    python scripts/reprocess_tune_folder.py "D:\\User Data\\Liam\\ActiveModeMap\\Demo1" \
        --band 330e3 470e3 --rank 5 --label modeA

Runs anywhere (no Igor, no Windows).
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from activemodemap.asylum import read_tune_txt, tune_to_complex   # noqa: E402
from activemodemap.lowrank import (reconstruct_map, dns_from_map,               # noqa: E402
                                   resonance_index, band_mask, dns_branch)

_XPAT = re.compile(r'_X(\d+)_')


def load_folder(folder, pattern="Tune_*.txt"):
    """Return (x_um sorted, freq_Hz, Z complex (npos, nfreq)) from a tune folder.

    Position is taken from the ``_X<nanometres>_`` field in the filename, which
    is how `AsylumInstrument` labels them.
    """
    paths = sorted(glob.glob(os.path.join(folder, pattern)))
    if not paths:
        raise SystemExit(f"no files matching {pattern} in {folder}")
    xs, raw = [], []
    for p in paths:
        m = _XPAT.search(os.path.basename(p))
        if not m:
            print(f"  skipping (no _X<nm>_ field): {os.path.basename(p)}")
            continue
        # tune_to_complex drops the NaN padding Igor leaves outside the swept
        # range, so the surviving grids can differ by a point or two between
        # files. Harmonise at the end rather than demanding exact equality.
        f, Z = tune_to_complex(read_tune_txt(p))
        xs.append(int(m.group(1)) / 1000.0)                 # nm -> um
        raw.append((f, Z))
    if not xs:
        raise SystemExit("no files had a parseable _X<nm>_ position field")

    F0 = raw[0][0]
    lo = max(float(r[0].min()) for r in raw)
    hi = min(float(r[0].max()) for r in raw)
    span0 = float(F0[-1] - F0[0]) or 1.0
    edge = max(max(abs(float(r[0][0]) - float(F0[0])),
                   abs(float(r[0][-1]) - float(F0[-1]))) for r in raw)
    if {r[0].size for r in raw} != {F0.size} or edge > 0:
        print(f"  grids differ ({sorted({r[0].size for r in raw})} points); "
              f"common {lo / 1e3:.1f}-{hi / 1e3:.1f} kHz, edges drift "
              f"{edge / 1e3:.2f} kHz ({100 * edge / span0:.1f}% of span)")
        if edge / span0 > 0.02:
            print("  ** the tune window MOVED between positions -- resonance "
                  "tracking was probably on. These spectra are not comparable "
                  "position to position. **")
    F = F0[(F0 >= lo - 1e-9) & (F0 <= hi + 1e-9)]
    if F.size < 8:
        raise SystemExit(f"only {F.size} frequency points common to all files")
    Zs = [Z if (f.shape == F.shape and np.allclose(f, F))
          else np.interp(F, f, Z.real) + 1j * np.interp(F, f, Z.imag)
          for f, Z in raw]
    o = np.argsort(xs)
    return np.asarray(xs)[o], F, np.asarray(Zs)[o]


def describe(F, Z):
    df = float(np.median(np.diff(F)))
    A = np.abs(Z).max(0)
    pk = int(A.argmax())
    half = A[pk] / np.sqrt(2)
    idx = np.where(A > half)[0]
    fwhm = (F[idx[-1]] - F[idx[0]]) if idx.size > 1 else np.nan
    print(f"  window   : {F[0] / 1e3:.1f} - {F[-1] / 1e3:.1f} kHz, "
          f"{F.size} points, {df:.0f} Hz step")
    print(f"  strongest: {F[pk] / 1e3:.2f} kHz, FWHM {fwhm / 1e3:.2f} kHz "
          f"= {fwhm / df:.1f} points, Q ~ {F[pk] / fwhm:.0f}")
    if fwhm / df < 5:
        print("  ** the resonance is undersampled (<5 points across the FWHM); "
              "narrow the Igor tune window or raise the point count **")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder")
    ap.add_argument("--pattern", default="Tune_*.txt")
    ap.add_argument("--band", nargs=2, type=float, metavar=("LO_HZ", "HI_HZ"),
                    help="analysis band in Hz; bracket ONE eigenmode's "
                         "resonance + antiresonance")
    ap.add_argument("--rank", type=int, default=5)
    ap.add_argument("--span", type=float, default=None,
                    help="position grid span in um (default: max measured x)")
    ap.add_argument("--step", type=float, default=1.0)
    ap.add_argument("--nboot", type=int, default=400)
    ap.add_argument("--label", default="reprocessed")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    xs, F, Z = load_folder(a.folder, a.pattern)
    print(f"\n{len(xs)} positions: {np.round(xs, 1)}")
    describe(F, Z)

    if a.band:
        m = band_mask(F, a.band)
        if m.sum() < 8:
            raise SystemExit(f"band keeps only {int(m.sum())} points")
        F, Z = F[m], Z[:, m]
        print(f"\nrestricted to {a.band[0] / 1e3:.0f}-{a.band[1] / 1e3:.0f} kHz "
              f"({F.size} points)")
    else:
        print("\n** no --band given: dns_from_map searches for the "
              "antiresonance within a FRACTION OF THE FULL SPAN, so a wide "
              "multi-mode window will give a meaningless D-NS. **")

    span = a.span if a.span is not None else float(xs.max())
    x_grid = np.arange(0.0, span + 1e-9, a.step)
    sel = [int(np.argmin(np.abs(x_grid - x))) for x in xs]
    if len(set(sel)) != len(sel):
        raise SystemExit("two positions fall on the same grid point; use a finer --step")

    rank = min(a.rank, len(sel))
    rec = reconstruct_map(x_grid, sel, Z, rank)
    ires = resonance_index(F, Z)
    dns = dns_from_map(x_grid, rec["Zrec"], F, ires)

    rng = np.random.default_rng(a.seed)
    noise = float(np.median(rec["sig"]))
    B, Bs = rec["B"], rec["B"][sel]
    samples = []
    for _ in range(a.nboot):
        zb = Z + noise * (rng.standard_normal(Z.shape)
                          + 1j * rng.standard_normal(Z.shape)) / np.sqrt(2)
        cb = np.linalg.lstsq(Bs, zb, rcond=None)[0]
        v = dns_from_map(x_grid, B @ cb, F, ires, guess_um=dns)
        if np.isfinite(v):
            samples.append(v)
    samples = np.asarray(samples)
    ci = (np.percentile(samples, 97.5) - np.percentile(samples, 2.5)
          if samples.size else np.nan)

    crossings, fa = dns_branch(x_grid, rec["Zrec"], F, ires)
    print(f"\nresonance used : {F[ires] / 1e3:.2f} kHz")
    print(f"notch defined  : at {int(np.isfinite(fa).sum())}/{x_grid.size} grid "
          "positions (elsewhere the antiresonance is outside the search window)")
    print(f"all crossings  : {np.round(crossings, 1)} um")
    if crossings.size > 1:
        print("  ** more than one branch crossing: this mode has more than one "
              "null spot in the span, so a single D-NS number is ambiguous. "
              "Inspect the map and pick the branch you mean. **")
    print(f"D-NS           : {dns:.1f} um   (95% CI {ci:.1f} um, "
          f"{len(xs)} positions, rank {rank})")
    print("NOTE: this CI is a bootstrap over the residual noise scale only. "
          f"With {len(xs)} positions and rank {rank} there are "
          f"{max(len(xs) - rank, 1)} residual dof, so it understates the true "
          "uncertainty — treat it as a lower bound.")

    out = os.path.join(a.folder, f"ActiveModeMap_{a.label}.npz")
    np.savez(out, x_grid=x_grid, freq=F, Zrec=rec["Zrec"], std=rec["std"],
             measured_x=xs, measured_Z=Z, dns=dns, dns_ci=ci,
             dns_samples=samples, ires=ires, rank=rank)
    print(f"saved {out}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        f_k = F / 1e3
        ext = [x_grid[0], x_grid[-1], f_k[0], f_k[-1]]
        fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
        amp = np.abs(rec["Zrec"]).T
        ax[0].imshow(np.log10(amp + 1e-12), origin="lower", aspect="auto",
                     extent=ext, cmap="viridis")
        for xv in xs:
            ax[0].axvline(xv, color="w", lw=0.7, alpha=0.85)
        ax[0].axvline(dns, color="#e34948", ls="--", lw=1.4)
        ax[0].set_title(f"mode shape, {len(xs)} positions")
        ax[1].imshow((rec["std"] / (amp.max() + 1e-12)).T, origin="lower",
                     aspect="auto", extent=ext, cmap="magma")
        ax[1].set_title("uncertainty (1σ, rel.)")
        for k in (0, 1):
            ax[k].set_xlabel("position (µm)"); ax[k].set_ylabel("frequency (kHz)")
        if samples.size:
            ax[2].hist(samples, bins=24, color="#2a78d6", alpha=0.8, density=True)
        ax[2].axvline(dns, color="k", lw=1.3)
        ax[2].set_title(f"D-NS = {dns:.1f} ± {ci / 2:.1f} µm")
        ax[2].set_xlabel("position (µm)")
        fig.tight_layout()
        png = os.path.join(a.folder, f"ActiveModeMap_{a.label}.png")
        fig.savefig(png, dpi=150, bbox_inches="tight")
        print(f"saved {png}")
    except Exception as e:
        print(f"(no figure: {e})")


if __name__ == "__main__":
    main()
