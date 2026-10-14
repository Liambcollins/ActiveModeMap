"""Two routes to the resonance enhancement E(x) used to calibrate a CR1 image (stiff probe B, R2).

Beam-model route (top): the blind Euler-Bernoulli fit sees only the CR1-CR3 frequencies, Q1 and the
shape of 1/InvOLS(x); E_model(x) is the ratio of its CR1 and quasi-static mode shapes.
Map route (bottom): the 8-position two-domain capture, P = (Z1 - Z2)/2, reconstructed (EB+GP) over the
CR1 and quasi-static bands; E_map(x) = |P_CR1| / |P_QS|.
Right: both against the measured two-domain E, and the CPD-corrected CR1 image at A divided by E_map(x_A).

    python tools/fig_e_routes_data.py   # once (writes results/fig_e_routes.npz)
    python tools/fig_e_routes.py [--out PATH] [--formats png,pdf,svg]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyArrowPatch
from matplotlib.colors import LinearSegmentedColormap, LogNorm
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import plotting as fp  # noqa: E402

from fmmpaper import style3 as S3, imgv3  # noqa: E402
OR, TE = S3.MODEL, S3.MAP              # beam-model lane, map lane (shared v3 palette)
SPOT = "#b2182b"
D1C, D2C = "#f4a582", "#92c5de"        # PPLN domains (as Fig. 1a)
D1L, D2L = S3.D1, S3.D2                # domain line colours
TEAL_MAP = S3.MAP_CMAP                 # spectral maps in coolwarm, as Figs 1-2


def mini(ax, xlabel=None, ylabel=None):
    ax.tick_params(length=2, pad=1.5)
    if xlabel: ax.set_xlabel(xlabel, labelpad=1)
    if ylabel: ax.set_ylabel(ylabel, labelpad=1)


def lever3d(ax, xsurv, L):
    """Oblique 3D view: chip, lever, tip, two-domain PPLN, laser spots at the survey positions."""
    P = lambda x, y, z: (x + 0.42 * y, z + 0.30 * y)            # oblique projection
    def box(x0, x1, y0, y1, z0, z1, top, side, front, lw=0.4):
        faces = [([(x0, y0, z1), (x1, y0, z1), (x1, y1, z1), (x0, y1, z1)], top),
                 ([(x0, y0, z0), (x1, y0, z0), (x1, y0, z1), (x0, y0, z1)], front),
                 ([(x1, y0, z0), (x1, y1, z0), (x1, y1, z1), (x1, y0, z1)], side)]
        for pts, c in faces:
            ax.add_patch(Polygon([P(*q) for q in pts], closed=True, facecolor=c, edgecolor="k", lw=lw, joinstyle="round"))
    ax.set_aspect("equal"); ax.axis("off")
    # sample: two domains
    box(-0.5, 5.3, -1.6, 2.2, -2.15, -1.75, D1C, "#d98a6a", "#e89878")
    box(5.3, 11.2, -1.6, 2.2, -2.15, -1.75, D2C, "#6d9fb8", "#7fb2cc")
    for xa, up in ((1.6, 1), (3.8, 1), (7.0, -1), (9.4, -1)):
        x_, y_ = P(xa, 0.3, -1.75 + 0.06)
        ax.annotate("", xy=(x_, y_ + (0.32 if up > 0 else 0.02)), xytext=(x_, y_ + (0.02 if up > 0 else 0.32)),
                    arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k", mutation_scale=5))
    # chip
    box(-2.0, 0.0, -0.9, 1.6, 0.0, 1.5, "#cfcfcf", "#9a9a9a", "#b5b5b5")
    # lever (shaded top face as strips)
    Lw, xL = 10.0, 0.0
    tip_x = xL + Lw - 0.45
    ax.add_patch(Polygon([P(xL, 0, 0.55), P(xL + Lw, 0, 0.55), P(xL + Lw, 0, 0.75), P(xL, 0, 0.75)], facecolor="#8f8f8f", edgecolor="k", lw=0.4))
    n = 24
    for k in range(n):
        a, b = xL + Lw * k / n, xL + Lw * (k + 1) / n; sh = 0.93 - 0.25 * k / n
        ax.add_patch(Polygon([P(a, 0, 0.75), P(b, 0, 0.75), P(b, 1.0, 0.75), P(a, 1.0, 0.75)], facecolor=(sh, sh, sh), edgecolor="none"))
    ax.add_patch(Polygon([P(xL, 0, 0.75), P(xL + Lw, 0, 0.75), P(xL + Lw, 1.0, 0.75), P(xL, 1.0, 0.75)], facecolor="none", edgecolor="k", lw=0.4))
    ax.add_patch(Polygon([P(xL + Lw, 0, 0.55), P(xL + Lw, 1.0, 0.55), P(xL + Lw, 1.0, 0.75), P(xL + Lw, 0, 0.75)], facecolor="#7a7a7a", edgecolor="k", lw=0.4))
    # tip (pyramid)
    apex = P(tip_x, 0.5, -1.72)
    for q1, q2, c in (((tip_x - 0.35, 0.15), (tip_x + 0.35, 0.15), "#6f6f6f"), ((tip_x + 0.35, 0.15), (tip_x + 0.35, 0.85), "#505050")):
        ax.add_patch(Polygon([P(q1[0], q1[1], 0.55), P(q2[0], q2[1], 0.55), apex], facecolor=c, edgecolor="k", lw=0.4))
    # laser spots at the 8 survey positions, beam on the 4th
    xs = xL + np.asarray(xsurv) / L * Lw
    for i, x in enumerate(xs):
        ax.plot(*P(x, 0.5, 0.76), "o", ms=2.6, color=SPOT, mec="w", mew=0.3, zorder=5)
    kb = 3; bx, by = P(xs[kb], 0.5, 0.76)
    ax.add_patch(Polygon([(bx - 0.12, by + 3.0), (bx + 0.12, by + 3.0), (bx + 0.02, by), (bx - 0.02, by)], facecolor=SPOT, alpha=0.55, edgecolor="none"))
    ax.add_patch(Rectangle((bx - 1.4, by + 3.0), 2.8, 0.8, facecolor="white", edgecolor="k", lw=0.5))
    ax.text(bx, by + 3.4, "laser", ha="center", va="center")
    ax.text(*P(xs[0] - 0.3, 0.5, 1.35), "$x_1$", ha="center", va="bottom", color=SPOT)
    ax.text(*P(xs[-1], 0.5, 1.35), "$x_8$", ha="center", va="bottom", color=SPOT)
    ax.text(*P(-1.0, 0.3, 2.15), "clamp", ha="center", va="bottom")
    x_, y_ = P(5.3, -1.6, -2.2); ax.text(x_, y_, "PPLN, two domains", ha="center", va="top")
    ax.set_xlim(-2.7, 12.3); ax.set_ylim(-3.2, 5.0)


def arrow(fig, xy0, xy1, color, lw=0.9, ms=7):
    fig.add_artist(FancyArrowPatch(xy0, xy1, transform=fig.transFigure, arrowstyle="-|>", mutation_scale=ms, lw=lw, color=color, shrinkA=0, shrinkB=0))


def line(fig, xs, ys, color, lw=0.9):
    fig.add_artist(plt.Line2D(xs, ys, transform=fig.transFigure, color=color, lw=lw, solid_capstyle="butt"))


def plot(out, formats):
    D = dict(np.load(F.config.RESULTS_DIR / "fig_e_routes_sem.npz", allow_pickle=True))
    S3.setup(); plt.rcParams.update({"savefig.bbox": None})
    W = fp.WIDTH_IN["double"]; H = 110 * fp.MM; fig = plt.figure(figsize=(W, H))
    A = lambda r: fig.add_axes(r)
    xl, L, xg = D["x_survey"], float(D["L_um"]), D["x_grid"]; f = D["freq_Hz"]
    jm = 3
    conv = None
    # ------------------------------------------------ a: acquisition
    ax = A([0.005, 0.56, 0.20, 0.37]); lever3d(ax, xl, L)
    fig.text(0.005, 0.975, "a", fontweight="bold", fontsize=8, va="top")
    fig.text(0.10, 0.975, "sparse two-domain capture", ha="center", va="top")
    ax = A([0.035, 0.08, 0.15, 0.37]); mini(ax, "frequency (MHz)")
    m = (f > 150e3) & (f < 2.0e6); cm = plt.get_cmap("Greys")
    for j in range(len(xl)):
        a = np.abs(D["Z_spot1"][j, m]); a = np.log10(np.maximum(a / a.max(), 1e-6))
        ax.plot(f[m] / 1e6, a / 4 + 0.62 * j, color=SPOT if j == jm else cm(0.45 + 0.5 * j / len(xl)), lw=0.45)
    ax.set_yticks([]); ax.set_xlim(0.15, 2.0); ax.spines[["left", "top", "right"]].set_visible(False)
    ax.text(-0.02, 0.0, "$x_1$", transform=ax.transAxes, ha="right", va="bottom"); ax.text(-0.02, 0.86, "$x_8$", transform=ax.transAxes, ha="right", va="bottom")
    ax.set_title("|Z| at 8 positions, 0 V (log)", pad=2)
    # ------------------------------------------------ c: beam-model route (lower lane; the map route leads as b)
    y0, h = 0.10, 0.215; w = 0.07; xs0 = [0.235, 0.325, 0.415]; xb, xms = 0.52, 0.65
    yhdr_model, yhdr_map = 0.485, 0.975
    fig.text(0.222, yhdr_model, "c", fontweight="bold", fontsize=8, va="top")
    fig.text(0.24, yhdr_model, "Model route", color=OR, fontweight="bold", va="top")
    fig.text(0.325, yhdr_model, "SEM geometry; PFM amplitudes withheld: the check", color="0.35", va="top")
    # three constraints
    ax = A([xs0[0], y0, w, h]); mini(ax, "f (MHz)")
    a = np.abs(D["Z_spot1"][jm]); ax.semilogy(f[m] / 1e6, a[m] / a[m].max(), color="0.25", lw=0.5)
    for fmod in D["cr_model_Hz"]:
        ax.axvline(fmod / 1e6, color=OR, lw=0.7, ls="--")
    ax.set_yticks([]); ax.set_xlim(0.15, 2.0); ax.set_ylim(None, 1.5); ax.set_title("frequencies", pad=3)
    b1 = D["bands_Hz"][0]; mb = (f > b1[0]) & (f < b1[1])
    ax = A([xs0[1], y0, w, h]); mini(ax, "f − f$_{CR1}$ (kHz)")
    a = np.abs(D["Z_spot1"][jm, mb]); i0 = int(np.argmax(a))
    ax.plot((f[mb] - f[mb][i0]) / 1e3, a / a.max(), "o", ms=1.2, color="0.25", mec="none")
    fl, al = D["Q_line_f"], D["Q_line_amp"]; il = int(np.argmax(al))
    ax.plot((fl - fl[il]) / 1e3, al / al.max(), color=OR, lw=0.8)
    ax.set_xlim(-4, 4); ax.set_xticks([-3, 0, 3]); ax.set_yticks([]); ax.set_title("Q", pad=3)
    ax.text(0.98, 0.97, f"{float(D['Q_meas']):.0f}", transform=ax.transAxes, ha="right", va="top")
    ax.text(0.98, 0.83, f"{float(D['Q_model']):.0f}", transform=ax.transAxes, ha="right", va="top", color=OR)
    ax = A([xs0[2], y0, w, h]); mini(ax, "x (µm)")
    nrm = D["static_model_grid"].max()
    ax.plot(xg, D["static_model_grid"] / nrm, color=OR, lw=0.8)
    ax.plot(D["x_static"], 1 / D["invols_static"] / nrm, "o", ms=2.4, color="k", mfc="w", mew=0.6)
    ax.set_yticks([]); ax.set_xlim(0, L); ax.set_xticks([0, 100, 200]); ax.set_title("1/InvOLS: check", pad=3, color="0.35")
    for sp in ax.spines.values(): sp.set_linestyle((0, (2, 2)))
    ax.text(0.97, 0.95, f"rms {100*float(D['static_check_rms']):.0f}%", transform=ax.transAxes, ha="right", va="top", color="0.35", fontsize=6)
    # bracket joining the two constraints into one fit
    yb = y0 + h + 0.065
    line(fig, [xs0[0] + w / 2, xs0[1] + w / 2], [yb, yb], OR, lw=0.8)
    for x_ in xs0[:2]:
        line(fig, [x_ + w / 2, x_ + w / 2], [yb, yb - 0.018], OR, lw=0.8)
    line(fig, [xs0[1] + w / 2, xb + 0.0475], [yb, yb], OR, lw=0.8); arrow(fig, (xb + 0.0475, yb), (xb + 0.0475, y0 + h + 0.038), OR, ms=5, lw=0.8)
    fig.text((xs0[0] + xs0[1] + w) / 2, yb + 0.006, "two constraints on one fit", ha="center", va="bottom", color=OR, fontsize=6.5)
    # beam model
    ax = A([xb, y0, 0.095, h]); ax.set_xlim(-0.8, 10.4); ax.set_ylim(-5.6, 2.2); ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values(): sp.set_color(OR); sp.set_linewidth(0.7)
    ax.set_title("EB fit, SEM geometry", pad=3)
    S3.boxes(ax)
    ax.add_patch(Rectangle((-0.6, -0.2), 0.6, 2.0, facecolor="#cfcfcf", edgecolor="k", lw=0.5, hatch="////"))
    ax.plot([0, 9.4], [0.8, 0.8], color="k", lw=1.8, solid_capstyle="butt")
    tx = 8.6; ax.add_patch(Polygon([[tx - 0.4, 0.75], [tx + 0.4, 0.75], [tx, -1.2]], facecolor="#6f6f6f", edgecolor="k", lw=0.4))
    zz = np.linspace(-1.2, -3.4, 9); ax.plot(tx + np.r_[0, 0.45 * (-1) ** np.arange(7), 0], zz, color=OR, lw=0.8)
    ax.plot([tx - 0.8, tx + 0.8], [-3.4, -3.4], color="k", lw=0.8); ax.text(tx + 0.7, -2.3, "k*", ha="left", va="center", color=OR)
    hx = np.linspace(tx - 2.4, tx, 9); ax.plot(hx, -1.2 + np.r_[0, 0.3 * (-1) ** np.arange(7), 0], color=OR, lw=0.7)
    ax.plot([tx - 2.4, tx - 2.4], [-1.8, -0.6], color="k", lw=0.8); ax.text(tx - 2.6, -1.2, "k$_{cone}$", ha="right", va="center", color=OR)
    ax.text(4.8, -4.0, "+ ζ;  L, set-back, tip height\nfixed from SEM", ha="center", va="top", color="0.3", fontsize=6.3, linespacing=1.1)
    arrow(fig, (xb + 0.095 + 0.004, y0 + h / 2), (xms - 0.008, y0 + h / 2), OR, ms=5, lw=0.7)
    # model transfer functions
    ax = A([xms, y0, w, h]); mini(ax, "x (µm)")
    zq, zc = D["z_qs_grid"], D["z_cr1_grid"]
    ax.plot(xg, zq / zq.max(), color="0.45", lw=0.8); ax.plot(xg, zc / zc.max(), color=OR, lw=0.9)
    ic = int(np.argmax(zc)); ax.text(xg[ic] - 8, 1.03, "|H$_{CR1}$|", color=OR, ha="center", va="bottom")
    ax.text(L * 1.0, 1.2, "|H$_{static}$|", color="0.4", ha="right", va="bottom")
    ax.set_yticks([]); ax.set_ylim(0, 1.45); ax.set_xlim(0, L); ax.set_xticks([0, 100, 200]); ax.set_title("model |H(x)|", pad=3)
    # ------------------------------------------------ b: map route (upper lane)
    y1, h1 = 0.655, 0.25; wc = 0.075
    fig.text(0.222, yhdr_map, "b", fontweight="bold", fontsize=8, va="top")
    fig.text(0.24, yhdr_map, "Map route", color=TE, fontweight="bold", va="top")
    fig.text(0.315, yhdr_map, "measured E at the 8 stops, interpolated by EB+GP: the calibration", color="0.35", va="top")
    ax = A([0.235, y1, wc, h1]); mini(ax, "f (kHz)")
    Z1, Z2 = D["Z_spot1"][jm, mb], D["Z_spot2"][jm, mb]; P = (Z1 - Z2) / 2; ph = np.exp(-1j * np.angle(P[np.argmax(np.abs(P))]))
    sc = np.abs(P).max(); fk = f[mb] / 1e3
    ax.plot(fk, (Z1 * ph).real / sc, color=D1L, lw=0.7); ax.plot(fk, (Z2 * ph).real / sc, color=D2L, lw=0.7); ax.plot(fk, (P * ph).real / sc, color=TE, lw=1.1)
    ax.axhline(0, color="0.6", lw=0.4); ax.set_yticks([]); ax.set_xlim(fk[i0] - 5, fk[i0] + 5); ax.set_ylim(-1.25, 1.25)
    ax.text(fk[i0] - 0.8, 0.95, "P$_{raw}$", color=TE, ha="right", va="center")
    ax.text(fk[i0] + 1.0, 0.5, "Z$_1$", color=D1L, ha="left", va="center"); ax.text(fk[i0] + 1.0, -0.6, "Z$_2$", color=D2L, ha="left", va="center")
    ax.set_title("P$_{raw}$=(Z$_1$−Z$_2$)/2", pad=3)
    # domain-CPD correction (QS channel along the lever)
    ax = A([0.36, y1, wc, h1]); mini(ax, "x (µm)", None)
    qr = np.abs(D["P_qs_raw"]).mean(1); qc = np.abs(D["P_qs_cor"]).mean(1); cv = D["d33_qs_stat"] / qc   # pm/V per unit |P| (InvOLS/32)
    ax.plot(xl, qr * cv, "o", ms=2.6, color="0.55", mfc="w", mew=0.6); ax.plot(xl, qc * cv, "o", ms=2.6, color=TE, mec="none")
    ax.axhline(float(D["d33_uniform"]), color=TE, lw=0.6, ls="--")
    ax.set_xlim(0, L); ax.set_xticks([0, 100, 200]); ax.set_ylim(6, 11); ax.set_yticks([7, 8, 9, 10]); ax.set_ylabel(S3.deff_label("QS"), labelpad=1)
    ax.text(0.04, 0.95, "raw", color="0.45", transform=ax.transAxes, va="top"); ax.text(0.04, 0.12, "corrected", color=TE, transform=ax.transAxes, va="bottom")
    ax.set_title("domain-CPD corr.", pad=3)
    fig.text(0.36 + wc / 2, y1 - 0.075, f"P = P$_{{raw}}$ − δ·b(x, f),  δ = {float(D['delta_V']):.2f} V", ha="center", color=TE, fontsize=6.3)
    ax = A([0.49, y1, 0.085, h1]); mini(ax, "x (µm)", "f (kHz)")
    Zm = np.abs(D["P_cr1_rec_cor"]); ff = D["f_cr1"] / 1e3
    ax.pcolormesh(xg, ff, Zm.T, cmap=TEAL_MAP, norm=LogNorm(vmin=Zm.max() / 300, vmax=Zm.max()), shading="nearest", rasterized=True)
    for x in xl:
        ax.plot(x, ff[-1], "v", ms=3, color=SPOT, mec="w", mew=0.3, clip_on=False, zorder=5)
    ax.set_xlim(0, L); ax.set_xticks([0, 100, 200]); ax.set_title("P map, CR1 (EB+GP)", pad=5); ax.set_ylim(ff[0], ff[-1])
    # E_map: measured stops, interpolant, leave-one-out, imaging coordinate
    ax = A([0.60, y1, 0.085, h1]); mini(ax, "x (µm)", None)
    ax.semilogy(xg, D["E_map_cor_grid"], color=TE, lw=1.0)
    ax.semilogy(xl, D["E_meas_cor"], "o", ms=2.8, color=TE, mec="none")
    ax.semilogy(xl, D["E_loo_cor"], "o", ms=3.2, color=TE, mfc="w", mew=0.6)
    xA_ = float(D["x_A"]); ax.axvline(xA_, color="k", lw=0.5, ls="--"); ax.text(xA_ + 4, 560, "A", fontsize=6.5)
    ax.set_xlim(0, L); ax.set_xticks([0, 100, 200]); ax.set_ylim(15, 900); ax.set_yticks([20, 50, 100, 200, 500]); ax.set_yticklabels(["20", "50", "100", "200", "500"]); ax.minorticks_off()
    ax.set_title("E$_{map}$(x)", pad=5, color=TE)
    ax.text(0.04, 0.05, "● stops  ○ left-out", transform=ax.transAxes, color=TE, fontsize=6.0, va="bottom")
    arrow(fig, (0.314, y1 + h1 / 2), (0.33, y1 + h1 / 2), TE, ms=5, lw=0.7)
    arrow(fig, (0.439, y1 + h1 / 2), (0.452, y1 + h1 / 2), TE, ms=5, lw=0.7); arrow(fig, (0.579, y1 + h1 / 2), (0.595, y1 + h1 / 2), TE, ms=5, lw=0.7)
    fig.text(0.694, y1 + h1 * 0.55, "$\\dfrac{|P_{CR1}|}{|P_{QS}|}$", color=TE, va="center")
    # ------------------------------------------------ d: E(x); e: matched validation; f: image
    xr, wr = 0.845, 0.145
    fig.text(0.79, 0.975, "d", fontweight="bold", fontsize=8, va="top")
    ax = A([xr, 0.665, wr, 0.255]); mini(ax, "x (µm)", "E")
    ax.semilogy(xg, D["E_model_grid"], color=OR, lw=1.0); ax.semilogy(xg, D["E_map_cor_grid"], color=TE, lw=1.0)
    ax.semilogy(xl, D["E_meas_cor"], "o", ms=2.6, color="k", mfc="w", mew=0.6)
    ax.axhline(float(D["Q_meas"]), color="0.5", lw=0.6, ls=":"); ax.text(L * 0.98, float(D["Q_meas"]) * 1.12, "Q", ha="right", va="bottom", color="0.4", fontsize=6.5)
    xA = float(D["x_A"]); ax.axvline(xA, color="k", lw=0.5, ls="--"); ax.text(xA + 4, 620, "A")
    ax.set_xlim(0, L); ax.set_ylim(15, 900); ax.set_xticks([0, 100, 200])
    ax.set_yticks([20, 50, 100, 200, 500]); ax.set_yticklabels(["20", "50", "100", "200", "500"]); ax.minorticks_off()
    ax.text(0.08, 0.30, "E$_{model}$", color=OR, transform=ax.transAxes); ax.text(0.08, 0.17, "E$_{map}$", color=TE, transform=ax.transAxes)
    ax.text(0.08, 0.04, "measured", color="k", transform=ax.transAxes)
    ax.set_title("E(x), both routes", pad=3)
    fig.text(0.79, 0.545, "e", fontweight="bold", fontsize=8, va="top")
    ax = A([xr, 0.37, wr, 0.15]); mini(ax, "x (µm)", S3.deff_label())
    ax.axhline(float(D["d33_uniform"]), color="0.5", lw=0.6, ls="--")
    ax.plot(xl, D["d33_qs_stat"], "o", ms=2.6, color="k", mfc="w", mew=0.6)
    ax.plot(xl, D["d33_cr_stat"], "s", ms=2.4, color=TE, mec="none")
    ax.set_xlim(0, L); ax.set_xticks([0, 100, 200]); ax.set_ylim(6, 9); ax.set_yticks([7, 8])
    ax.text(0.03, 0.95, "QS", transform=ax.transAxes, va="top"); ax.text(0.03, 0.06, "CR1 ÷ E$_{map}$, left out", color=TE, transform=ax.transAxes, va="bottom", fontsize=6.3)
    ax.text(0.97, 0.95, f"{np.median(D['d33_qs_stat']):.2f} / {np.median(D['d33_cr_stat']):.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=6.3)
    ax.set_title("stationary check", pad=3)
    fig.text(0.79, 0.245, "f", fontweight="bold", fontsize=8, va="top")
    IMA = imgv3.position_A(float(D["E_map_cor_A"]))["1V"]; img = IMA["cr"]
    wi = 0.075; hi = wi * W / H; yi = 0.235 - hi
    ax = A([0.815, yi, wi, hi]); h_ = S3.sym_im(ax, img, 10); S3.boxes(ax)
    ax.set_xticks([]); ax.set_yticks([])
    npx = img.shape[1]; sb = 2.0 / float(D["img_scan_um"]) * npx
    ax.plot([npx * 0.06, npx * 0.06 + sb], [img.shape[0] * 0.07] * 2, color="k", lw=1.4); ax.text(npx * 0.06 + sb / 2, img.shape[0] * 0.1, "2 µm", color="k", ha="center", va="bottom", fontsize=6)
    cax = fig.add_axes([0.815 + wi + 0.006, yi, 0.006, hi]); cb = fig.colorbar(h_, cax=cax, ticks=[-10, 0, 10]); cb.set_label("pm V$^{-1}$", labelpad=1); cb.ax.tick_params(length=1.5, pad=1)
    fig.text(0.815, 0.242, "CR1 image at A ÷ E$_{map}$(x$_A$)", ha="left", va="bottom")
    ratio = float(D["E_image_A"]) / float(D["E_survey_A"])
    md = IMA["cr_med"]; D["img_d1"], D["img_d2"] = md
    tx_ = [f"domains: {md[0]:.1f} / {md[1]:.1f} pm V$^{{-1}}$",
           f"scan: E$_{{image}}$ = {ratio:.2f} E$_{{survey}}$", f"→ {np.mean(md) / ratio:.1f} (survey {float(D['d33_uniform']):.2f})"]
    for k, t_ in enumerate(tx_):
        fig.text(0.815, yi - 0.012 - 0.03 * k, t_, ha="left", va="top", fontsize=6.5, color="k" if k == 0 else "0.35")
    # ------------------------------------------------ connectors
    xsplit = 0.205
    line(fig, [0.19, xsplit], [0.5, 0.5], "k", lw=0.8)
    line(fig, [xsplit, xsplit], [0.5, 0.94], TE); line(fig, [xsplit, 0.218], [0.94, 0.94], TE)
    line(fig, [xsplit, xsplit], [0.45, 0.5], OR); line(fig, [xsplit, 0.218], [0.45, 0.45], OR)
    fig.add_artist(plt.Line2D([xsplit], [0.5], transform=fig.transFigure, marker="o", ms=3, color="k"))
    ym = y0 + h * 0.5; xm = 0.78
    line(fig, [xms + w + 0.004, xm], [ym, ym], OR); line(fig, [xm, xm], [ym, 0.70], OR); arrow(fig, (xm, 0.70), (xr - 0.055, 0.70), OR, ms=5)
    fig.text(xms + w + 0.008, ym + 0.012, "$E_{model}(x)$", color=OR, va="bottom")
    fig.text(xms + w + 0.008, ym - 0.012, "$\\dfrac{|H_{CR1}|}{|H_{static}|}$", color=OR, va="top")
    yt = y1 + h1 * 0.55
    arrow(fig, (0.74, yt), (xr - 0.055, yt), TE, ms=5)
    out = Path(out)
    for ext in formats:
        p = out.with_suffix("." + ext); fig.savefig(p, dpi=600, bbox_inches=None); print("wrote", p)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=str(S3.OUT / "Fig3_two_routes.png")); ap.add_argument("--formats", default="png")
    a = ap.parse_args(); plot(a.out, tuple(a.formats.split(",")))
