"""Measurement-concept figure with real data (SCM-PIT R2, 500 nN, 0 V).

(a) oblique view: lever over two-domain PPLN, interferometric spot, the eight sparse laser positions
    (bias-survey / live-capture design) and the measured CR1 profile |Z(x, f_CR1)| drawn over the lever;
(b) tip-sample junction as modelled: normal contact spring k* with modal damping, tip-cone compliance
    k_cone in series, lateral coupling through tip height h and the 11 deg lever tilt;
(c, d) dense amplitude and phase maps Z(x, f) (161 positions, 1 um pitch), lever frame, 0 to L;
(e) top view of the two-domain spectroscopy (laser positions x_i on the lever, tip on spot 1 / spot 2);
(f) amplitude and phase spectra at three laser positions taken from the dense map.

    python tools/fig_schematic_v3.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, Circle, FancyArrowPatch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import style3 as S, spectra, eb_fit, ebfit, physrec  # noqa: E402

R = F.config.RESULTS_DIR
PROBE = sys.argv[1] if len(sys.argv) > 1 else "stiff"
if PROBE == "stiff":      # SCM-PIT lever B, run R2: frame from the InvOLS ruler, tip set-back from the blind fit
    X0 = 28.1; L = 226.5
    c6 = json.load(open(R / "_cache_fig6.json"))["R2"]["central"]["summary"]; SETBACK = float(c6["setback_um"])
    live = json.load(open(R / "live_ebgp_d33_stiff_r2.json")); XS = np.array([r["x_lever_um"] for r in live["table"]])
    s = F.load("scmpitB_r2_dense"); FR = (240e3, 1910e3); NAME = "Fig1_schematic_measured"; SRC = "blind EB fit"
else:                     # PPP-CONTAu: frame, length and set-back from the calibrated lever (dense-map geometry fit)
    CAL = ebfit.load_calibration("soft", "mass"); X0 = CAL["P"]["c_um"]; L = CAL["P"]["L_um"]; SETBACK = CAL["P"]["setback_um"]
    s = F.load("ppp_dense_1um"); FR = (40e3, 1090e3); NAME = "Fig1_schematic_measured_soft"; SRC = "calibrated EB"
x = s.x_um - X0; f = s.freq_Hz; Z = s.Z[0]; bands = dict(s.probe.bands_Hz); XTIP = L - SETBACK
if PROBE != "stiff":
    from fmmpaper import recon
    SEL = np.array(recon.select_equispaced(x, 12)); XS = x[SEL]


def remove_delay(f, Z, bands, k=60):
    """Fit and remove the instrument phase delay: phase = a_seg - 2 pi f tau, fitted to the amplitude-weighted
    common phase (doubled-angle mean over x, so node sign flips cancel) in the gaps between resonances,
    with one offset per gap (each resonance passage adds -180 deg). Returns Z exp(i(2 pi f tau - a_0)), tau, rms."""
    m = f >= 100e3; ff = f[m]; n = (ff.size // k) * k; fd = ff[:n].reshape(-1, k).mean(1)
    Zd = Z[:, m][:, :n].reshape(Z.shape[0], -1, k).mean(2)
    psu = np.unwrap(np.angle(np.sum(np.abs(Zd) * np.exp(2j * np.angle(Zd)), 0))) / 2
    edges = [100e3] + [v for b in bands.values() for v in (b[0] - 15e3, b[1] + 15e3)] + [f.max()]
    segs = [(edges[2 * i], edges[2 * i + 1]) for i in range(len(edges) // 2)]
    rows, y = [], []
    for i, (a, b) in enumerate(segs):
        for j in np.flatnonzero((fd > a) & (fd < b)):
            r = np.zeros(len(segs) + 1); r[i] = 1; r[-1] = fd[j]; rows.append(r); y.append(psu[j])
    A_, y = np.array(rows), np.array(y); sol, *_ = np.linalg.lstsq(A_, y, rcond=None)
    tau = -sol[-1] / (2 * np.pi); rms = np.degrees(np.std(y - A_ @ sol))
    return Z * np.exp(1j * (2 * np.pi * f * tau - sol[0])), tau, rms


Z, TAU, RMS = remove_delay(f, Z, bands)
print(f"phase delay removed: tau = {TAU*1e9:.0f} ns (fit residual {RMS:.1f} deg)")
m = (f >= FR[0]) & (f <= FR[1])
dec = 4                                     # display decimation in frequency (block mean of the complex response)
nf = (m.sum() // dec) * dec; fz = f[m][:nf].reshape(-1, dec).mean(1); Zd = Z[:, m][:, :nf].reshape(len(x), -1, dec).mean(2)
A = 20 * np.log10(np.abs(Zd) / np.abs(Zd).max()); PH = np.degrees(np.angle(Zd))
QUIET = A < -66                              # phase undefined where the response sits near the noise floor
PHm = np.where(QUIET, np.nan, PH)
f1, prof1 = spectra.on_resonance_profile(f, Z, bands["CR1"]); j1 = int(np.argmin(np.abs(f - f1))); cr1 = np.abs(Z[:, j1]); cr1 /= cr1.max()
print(f"lever frame {x.min():.1f}-{x.max():.1f} um; tip at {XTIP:.1f} (set-back {SETBACK:.1f}); f_CR1 {f1/1e3:.1f} kHz; sparse x {np.round(XS,1)}")

S.setup(); fig = S.figure(112); W, H = S.W_DOUBLE, 112 * S.MM
# ------------------------------------------------------------------ (a) oblique view
ax = fig.add_axes([0.0, 0.42, 0.56, 0.56]); ax.set_aspect("equal"); ax.axis("off")
th = np.radians(11.0); sc = 10.0 / L                       # 10 drawing units = L
P = lambda u, y, z: (u * np.cos(th) * 1.0 + 0.45 * y, -u * np.sin(th) * 0.55 + z + 0.32 * y)   # lever coordinate u (drawing units), tilted down to the tip
Pf = lambda xx, y, z: (xx + 0.45 * y, z + 0.32 * y)        # flat (sample, chip)
def quad(pts, fc, ec="k", lw=0.4, **kw): ax.add_patch(Polygon(pts, closed=True, facecolor=fc, edgecolor=ec, lw=lw, joinstyle="round", **kw))
zs = -2.05                                                  # sample top
tipx_draw, tipz = P(XTIP * sc, 0.5, 0)
# sample: two domains along x, split under the lever's tip region
xb = tipx_draw - 1.2
for (x0_, x1_), c, cs in (((-1.5, xb), "#f4a582", "#d98a6a"), ((xb, 12.0), "#92c5de", "#6d9fb8")):
    quad([Pf(x0_, -1.6, zs), Pf(x1_, -1.6, zs), Pf(x1_, 2.4, zs), Pf(x0_, 2.4, zs)], c)
    quad([Pf(x0_, -1.6, zs - 0.4), Pf(x1_, -1.6, zs - 0.4), Pf(x1_, -1.6, zs), Pf(x0_, -1.6, zs)], cs)
quad([Pf(12.0, -1.6, zs - 0.4), Pf(12.0, 2.4, zs - 0.4), Pf(12.0, 2.4, zs), Pf(12.0, -1.6, zs)], "#5f8fae")
for xa, up in ((1.5, 1), (4.5, 1), (10.4, -1)):
    a0 = Pf(xa, 0.4, zs)
    ax.annotate("", xy=(a0[0], a0[1] + (0.4 if up > 0 else 0.02)), xytext=(a0[0], a0[1] + (0.02 if up > 0 else 0.4)), arrowprops=dict(arrowstyle="-|>", lw=0.6, color="k", mutation_scale=5))
# chip
quad([Pf(-2.2, -0.9, 0.0), Pf(0, -0.9, 0.0), Pf(0, -0.9, 1.6), Pf(-2.2, -0.9, 1.6)], "#b5b5b5")
quad([Pf(0, -0.9, 0.0), Pf(0, 1.8, 0.0), Pf(0, 1.8, 1.6), Pf(0, -0.9, 1.6)], "#9a9a9a")
quad([Pf(-2.2, -0.9, 1.6), Pf(0, -0.9, 1.6), Pf(0, 1.8, 1.6), Pf(-2.2, 1.8, 1.6)], "#d0d0d0")
# lever (tilted 11 deg), shaded top
Lw = L * sc; n = 30
for k in range(n):
    a_, b_ = Lw * k / n, Lw * (k + 1) / n; g = 0.92 - 0.22 * k / n
    quad([P(a_, 0, 0.2), P(b_, 0, 0.2), P(b_, 1, 0.2), P(a_, 1, 0.2)], (g, g, g), ec="none")
quad([P(0, 0, 0.2), P(Lw, 0, 0.2), P(Lw, 1, 0.2), P(0, 1, 0.2)], "none")
quad([P(0, 0, 0.0), P(Lw, 0, 0.0), P(Lw, 0, 0.2), P(0, 0, 0.2)], "#8a8a8a")
# tip at L - set-back, apex on the domain boundary side (domain 2 here)
ut = XTIP * sc; apex = (P(ut, 0.5, 0)[0], zs + 0.32 * 0.5)
quad([P(ut - 0.3, 0.2, 0), P(ut + 0.3, 0.2, 0), apex], "#6f6f6f"); quad([P(ut + 0.3, 0.2, 0), P(ut + 0.3, 0.8, 0), apex], "#505050")
# model mode shapes at CR1-CR3 (blind EB fit, R2): signed tip-driven displacement along the lever, drawn above it
if PROBE == "stiff":
    pb = np.array(json.load(open(R / "_cache_fig6.json"))["R2"]["central"]["p"])
    stb = eb_fit.ContactState("R2", np.zeros(3), 0.0, np.array([]), np.array([]), L_um=L)
    fkm = eb_fit.peaks(pb, stb); mm, rr = eb_fit._model(pb, stb, fkm); xm = mm.xi * L; MODES = list(zip(fkm, rr["piezo"]))
else:   # per-band contact state fitted to the 12 sparse spectra, evaluated over the whole lever
    xm = np.linspace(0, L, 400); MODES = []
    for b in bands:
        zb, fb = physrec.band_slice(f, Z, bands[b], n_max=250)
        o = ebfit.rec_eb2(s.x_um, SEL, zb[SEL], fb, CAL, b, x_eval=xm + X0, gp=False)
        jm = int(np.argmax(np.abs(o["Zeb"]).mean(0))); MODES.append((fb[jm], o["Zeb"][:, jm]))
MODE_C = [S.CR1, S.MODEL, S.MAP, S.POLY, S.QS]
NM = len(MODES); DZ = 0.95 if NM <= 3 else 0.62; AMP = 0.38 if NM <= 3 else 0.26
for k_, (fm, zr) in enumerate(MODES):
    sh = np.real(zr * np.exp(-1j * np.angle(zr[np.argmax(np.abs(zr))]))); sh = sh / np.abs(sh).max()
    base = 0.6 + DZ * k_; keep = xm <= XTIP
    ax.plot(*zip(*[P(u, 0.5, base) for u in (0, XTIP * sc)]), color="0.75", lw=0.5, ls=(0, (2, 2)))
    ax.plot(*zip(*[P(u, 0.5, base + AMP * v) for u, v in zip(xm[keep] * sc, sh[keep])]), color=MODE_C[k_], lw=1.1)
    lx, ly = P(XTIP * sc, 0.5, base); ax.text(lx + 0.15, ly, f"CR{k_+1}, {fm/1e3:.0f} kHz", color=MODE_C[k_], va="center", ha="left", fontsize=6.5)
lx, ly = P(XTIP * sc, 0.5, 0.6 + DZ * NM); ax.text(lx + 0.15, ly - 0.2, "model mode shapes", va="center", ha="left", fontsize=6.5)
# sparse laser positions and the interferometer beam
for i, xi in enumerate(XS):
    ax.plot(*P(xi * sc, 0.5, 0.21), "o", color=S.CR1 if False else "#b2182b", ms=3, mec="white", mew=0.3, zorder=6)
bx, by = P(XS[3] * sc, 0.5, 0.21)
ax.add_patch(Polygon([(bx - 0.13, by + 4.0), (bx + 0.13, by + 4.0), (bx + 0.02, by), (bx - 0.02, by)], facecolor="#e34a33", alpha=0.6, edgecolor="none"))
ax.add_patch(Rectangle((bx - 0.6, by + 4.0), 1.2, 0.9, facecolor="#9a9a9a", edgecolor="k", lw=0.4))
ax.text(bx + 0.8, by + 4.45, "interferometric\nreadout (QPDI)", va="center", ha="left", linespacing=1.0)
ax.text(*P(XS[0] * sc, 0.0, -0.15), "$x_1$", ha="center", va="top", color="#b2182b"); ax.text(*P(XS[-1] * sc, 0.0, -0.15), f"$x_{{{len(XS)}}}$", ha="right", va="top", color="#b2182b")
# axis annotations: x = 0 at the clamp, tip at L - set-back, 11 deg tilt
o = P(0, 1, 0.2); ax.text(o[0] + 0.05, o[1] + 0.1, "x = 0", ha="center", va="bottom")
tp = P(Lw, 1.0, 0.2); ax.text(tp[0] + 0.15, tp[1] - 0.05, "x = L", ha="left", va="top")
ax.text(apex[0] + 0.2, apex[1] + 0.15, f"tip, L − {SETBACK:.0f} µm (fit)", ha="left", va="bottom")
ax.plot([Pf(0, -0.9, -0.05)[0], Pf(2.6, -0.9, -0.05)[0]], [Pf(0, -0.9, -0.05)[1]] * 2, color="0.4", lw=0.5, ls="--")
ax.text(Pf(2.75, -0.9, -0.05)[0], Pf(2.75, -0.9, -0.05)[1] - 0.05, "11°", va="center")
ax.text(*Pf(-1.1, -0.9, 1.75), "clamp", ha="center", va="bottom")
ax.text(*Pf(0.0, -1.6, zs - 0.55), "PPLN: domain 1", ha="left", va="top", color="#a8463a")
ax.text(*Pf(12.0, -1.6, zs - 0.55), "domain 2", ha="right", va="top", color="#2f6d97")
ax.set_xlim(-2.6, 13.6); ax.set_ylim(-3.4, 6.1)
S.flabel(fig, 0.0, 0.985, "a")
# ------------------------------------------------------------------ (b) junction inset
ax = fig.add_axes([0.4, 0.05, 0.17, 0.32]); ax.set_xlim(-1.4, 1.6); ax.set_ylim(-2.0, 2.6); ax.set_aspect("equal"); ax.axis("off")
ax.add_patch(Circle((0.1, 0.1), 1.95, facecolor="white", edgecolor="0.3", lw=0.6))
quad([(-0.55, 1.6), (0.55, 1.6), (0.0, 0.25)], "#7a7a7a")
zz = np.linspace(0.25, -1.1, 11); ax.plot(np.r_[0, 0.22 * (-1) ** np.arange(9), 0] * 1.0 - 0.2, zz, color=S.INK, lw=0.8)
ax.plot([-0.55, 0.15], [-1.1, -1.1], color=S.INK, lw=0.8)
ax.plot([0.25, 0.25], [0.25, -0.25], color=S.INK, lw=0.8); ax.add_patch(Rectangle((0.12, -0.75), 0.26, 0.35, facecolor="white", edgecolor=S.INK, lw=0.7))
ax.plot([0.25, 0.25], [-0.75, -1.1], color=S.INK, lw=0.8); ax.plot([0.18, 0.32], [-0.45, -0.45], color=S.INK, lw=0.8)
hx = np.linspace(0.0, 1.05, 9); ax.plot(hx, 0.3 + np.r_[0, 0.13 * (-1) ** np.arange(7), 0], color=S.INK, lw=0.7); ax.plot([1.05, 1.05], [0.05, 0.55], color=S.INK, lw=0.8)
ax.text(-0.5, -0.45, "k*", ha="right", va="center"); ax.text(0.45, -0.58, "ζ", ha="left", va="center")
ax.text(0.55, 0.62, "lateral (h, 11°)", ha="center", va="bottom", fontsize=6.3)
ax.text(-0.55, 1.15, "k$_{cone}$", ha="right", va="center")
ax.add_patch(Rectangle((-1.3, -1.6), 2.9, 0.5, facecolor=S.D2, edgecolor="none", alpha=0.6, zorder=0))
ax.text(0.1, 2.12, "tip–sample junction (model)", ha="center", va="bottom", fontsize=6.3)
S.flabel(fig, 0.39, 0.415, "b")
# ------------------------------------------------------------------ (c, d) maps over 0..L
def maprow(rect, Zm, cmap, vmin, vmax, cbl, letter, ticks=None):
    ax = fig.add_axes(rect); S.boxes(ax)
    ax.add_patch(Rectangle((0, fz[0] / 1e3), L, (fz[-1] - fz[0]) / 1e3, facecolor="#f0f0f0", edgecolor="none", zorder=0))
    ax.add_patch(Rectangle((x.min(), fz[0] / 1e3), np.ptp(x), (fz[-1] - fz[0]) / 1e3, facecolor="#cfcfcf", edgecolor="none", zorder=0))
    h = ax.pcolormesh(x, fz / 1e3, Zm.T, cmap=cmap, vmin=vmin, vmax=vmax, shading="nearest", rasterized=True)
    for xi in XS: ax.axvline(xi, color="k", lw=0.5, ls=(0, (2, 2)), alpha=0.7)
    ax.plot(XS, np.full(len(XS), fz[-1] / 1e3), "v", color="#b2182b", ms=3.2, mec="white", mew=0.3, clip_on=False, zorder=6)
    ax.axvline(XTIP, color="k", lw=0.8)
    for b in bands: ax.text(x.min() + 2, np.mean(bands[b]) / 1e3 + (0.027 if b != list(bands)[-1] else -0.03) * np.ptp(fz) / 1e3, b, ha="left", va="bottom" if b != list(bands)[-1] else "top", fontsize=6.5, bbox=dict(fc="white", ec="none", alpha=0.75, pad=0.6))
    ax.set_xlim(0, L); ax.set_ylim(fz[0] / 1e3, fz[-1] / 1e3); ax.set_ylabel("frequency (kHz)")
    cax = fig.add_axes([rect[0] + rect[2] + 0.008, rect[1], 0.008, rect[3]]); cb = fig.colorbar(h, cax=cax, ticks=ticks); cb.set_label(cbl, labelpad=1)
    S.flabel(fig, rect[0] - 0.06, rect[1] + rect[3] + 0.04, letter); return ax
axc = maprow([0.625, 0.565, 0.31, 0.37], A, S.MAP_CMAP, -60, 0, "|Z| (dB re max)", "c")
axc.set_xticklabels([]); axc.set_title(f"amplitude |Z(x, f)|, dense map ({len(x)} positions, 1 µm)", pad=9)
axd = maprow([0.625, 0.115, 0.31, 0.37], PHm, "twilight_shifted", -180, 180, "arg Z (deg)", "d", ticks=[-180, -90, 0, 90, 180])
axd.set_xlabel("laser position x from clamp (µm)"); axd.set_title(f"phase arg Z(x, f), {TAU*1e9:.0f} ns delay removed; grey: |Z| < −66 dB", pad=9)
for a_ in (axc, axd):
    a_.text(XTIP - 1.5, np.mean(fz) / 1e3, "tip", ha="right", va="center", fontsize=6.5, rotation=90)
    a_.text(x.min() / 2, np.mean(fz) / 1e3 + 0.05 * np.ptp(fz) / 1e3, "not\nmapped", ha="center", va="center", color=S.GREY, fontsize=6.5)
# ------------------------------------------------------------------ (e, f) spectra at three positions
pick = [0, 3, 6] if len(XS) <= 8 else [0, 5, 10]; cols = ["#bdbdbd", "#7a7a7a", "#222222"]
ja = [int(np.argmin(np.abs(x - XS[k]))) for k in pick]
axa = fig.add_axes([0.065, 0.25, 0.31, 0.15]); axp = fig.add_axes([0.065, 0.075, 0.31, 0.15])
for k, j, c in zip(pick, ja, cols):
    axa.semilogy(fz / 1e3, np.abs(Zd[j]) / np.abs(Zd).max(), color=c, lw=0.7)
    axp.plot(fz / 1e3, PHm[j], ".", color=c, ms=0.8, mec="none", rasterized=True)
    axa.text(0.47, 0.97 - 0.14 * pick.index(k), f"$x_{{{k+1}}}$ = {x[j]:.0f} µm", transform=axa.transAxes, ha="left", va="top", color=c)
axa.set_xticks(axp.get_xticks() if False else ([500, 1000, 1500] if PROBE == "stiff" else [200, 400, 600, 800, 1000])); axa.set_xlim(fz[0] / 1e3, fz[-1] / 1e3); axp.set_xlim(fz[0] / 1e3, fz[-1] / 1e3); axa.set_xticklabels([])
axa.set_ylabel("|Z| (norm.)"); axp.set_ylabel("arg Z (deg)"); axp.set_yticks([-180, 0, 180]); axp.set_ylim(-200, 200)
axp.set_xlabel("frequency (kHz)"); axp.set_xticks([500, 1000, 1500] if PROBE == "stiff" else [200, 400, 600, 800, 1000])
for b in bands:
    for a_ in (axa, axp): a_.axvspan(bands[b][0] / 1e3, bands[b][1] / 1e3, color="#f2f2f2", lw=0, zorder=0)
    axa.text(np.mean(bands[b]) / 1e3, 1.6, b, ha="center", va="bottom", fontsize=6.5)
axa.set_ylim(None, 1.5)
S.flabel(fig, 0.0, 0.415, "e")
S.save(fig, NAME)
