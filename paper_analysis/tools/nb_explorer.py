from nbbuild import SETUP

CELLS = [
("md", """
# Reconstruction explorer

Interactive companion to Figs. 2–3. Pick a probe, a band, how many positions N and where they sit,
and compare the reconstruction arms against the dense map: the map itself, the spectrum at any
position, the on-resonance mode shape with its nodes, and the held-out error per position.

- **GP** and **low-rank** run instantly on every change.
- **EB** and **EB+GP** use the calibrated-lever fitter (`fmmpaper.ebfit`, ~1 s per band): lever geometry
  is fixed by a one-time joint calibration (`results/lever_calibration_*.json`); per band only k*/k, ζ, ε
  and a complex gain are fitted from the selected spectra. With band = **full** every CR band is fitted
  and stitched into the wideband map (EB is undefined between bands); errors are then scored on the
  CR bands only, for every arm, and one mode-shape panel is drawn per band.
- Positions are in the lever frame, x − x₀ (stiff x₀ = 28.1 µm, soft clamp at stage −19.0 µm).

Run all cells, then use the controls under the last cell. Needs `ipywidgets` (`pip install ipywidgets`).
"""),
("code", SETUP),
("code", """
import json, hashlib, time
import ipywidgets as W
from IPython.display import display, clear_output
from fmmpaper import physrec, ebfit

X0 = {"stiff": 28.1, "soft": -19.0}
CAL_TAG = {"stiff": "mass", "soft": "mass"}
KEYS = {"stiff": ("scmpitB_r2_dense", "scmpitB", 2000e3), "soft": ("ppp_dense_1um", "pppcontau", 1100e3)}
DATA = {}
def load(probe):
    if probe not in DATA:
        k, g, fmax = KEYS[probe]; s = F.load(k)
        DATA[probe] = dict(s=s, x=s.x_um - X0[probe], x_stage=s.x_um, f=s.freq_Hz, Z=s.Z[0], geom=physrec.GEOM[g], fmax=fmax,
                           bands=dict(s.probe.bands_Hz), cal=ebfit.load_calibration(probe, CAL_TAG[probe]))
    return DATA[probe]

def eb_fit(D, band, sel, zb, fb):
    t0 = time.time(); out = ebfit.rec_eb2(D["x_stage"], sel, zb[sel], fb, D["cal"], band)
    th = out["theta"]
    print(f"EB fit {time.time()-t0:.1f} s: k*/k {th['k_ratio']:.0f}, zeta {th['zeta']:.4f}, eps {th['eps']:.3f}, chi2 {th['red_chi2']:.1f}")
    return out

def design(x, N, mode, seed, custom):
    if mode == "custom" and custom.strip():
        want = np.array([float(v) for v in custom.replace(";", ",").split(",") if v.strip()])
        return sorted({int(np.argmin(np.abs(x - w))) for w in want})
    if mode == "random":
        rng = np.random.default_rng(int(seed)); return sorted(rng.choice(len(x), N, replace=False).tolist())
    return recon.select_equispaced(x, N)
"""),
("code", """
ARM_STYLE = {"GP": (fp.C["gp"], ":"), "low-rank": (fp.C["lowrank"], "--"), "EB": (fp.C["eb"], "-"), "EB+GP": (fp.C["ebgp"], "-")}
STATE = {"eb": {}}      # eb results keyed by (probe, band, tuple(sel))

def _interp_rows(f_to, f_from, M):
    re = np.apply_along_axis(lambda r: np.interp(f_to, f_from, r), 1, M.real)
    im = np.apply_along_axis(lambda r: np.interp(f_to, f_from, r), 1, M.imag)
    return re + 1j * im

def compute(probe, band, N, mode, seed, custom, rank, arms, run_eb=True):
    D = load(probe); x, f, Z = D["x"], D["f"], D["Z"]
    sel = design(x, N, mode, seed, custom); h = recon.held_out(len(x), sel)
    bands = list(D["bands"]) if band == "full" else [band]
    if band == "full":
        zb, fb = Z, f
    else:
        zb, fb = physrec.band_slice(f, Z, D["bands"][band], n_max=250)
    # frequency mask of each CR band on the working grid (all of it for a single band)
    masks = {b: (fb >= D["bands"][b][0]) & (fb <= D["bands"][b][1]) for b in bands} if band == "full" else {band: np.ones(fb.size, bool)}
    span = np.zeros(fb.size, bool)
    for m in masks.values(): span |= m
    rec = {}
    if "GP" in arms: rec["GP"] = recon.rec_gp(x, sel, zb[sel])["Zrec"]
    if "low-rank" in arms: rec["low-rank"] = recon.rec_lowrank(x, sel, zb[sel], rank=min(int(rank), len(sel)))["Zrec"]
    if "EB" in arms or "EB+GP" in arms:
        eb = np.full(zb.shape, np.nan + 0j); ebgp = eb.copy()
        for b in bands:
            k = (probe, b, tuple(sel))
            if k not in STATE["eb"]:
                zbb, fbb = (zb, fb) if band != "full" else physrec.band_slice(f, Z, D["bands"][b], n_max=250)
                STATE["eb"][k] = eb_fit(D, b, sel, zbb, fbb); STATE["eb"][k]["f"] = fbb
            o = STATE["eb"][k]; m = masks[b]
            if band == "full":
                eb[:, m] = _interp_rows(fb[m], o["f"], o["Zeb"]); ebgp[:, m] = _interp_rows(fb[m], o["f"], o["Zrec"])
            else:
                eb, ebgp = o["Zeb"], o["Zrec"]
        if "EB" in arms: rec["EB"] = eb
        if "EB+GP" in arms: rec["EB+GP"] = ebgp
    return dict(x=x, f=fb, Z=zb, sel=np.array(sel), h=h, rec=rec, band=band, probe=probe, bands=bands, masks=masks, span=span)

def _nrmse(zr, Z, span):
    return recon.nrmse(zr[..., span], Z[..., span])

def draw(R, show_arm, ipos):
    x, f, Z, sel, h, rec, span = R["x"], R["f"], R["Z"], R["sel"], R["h"], R["rec"], R["span"]
    nb = len(R["bands"]); full = R["band"] == "full"
    fig = plt.figure(figsize=(13, 12 if full else 8.5))
    gs = fig.add_gridspec(3, max(nb, 3), hspace=0.5, wspace=0.35, height_ratios=[1, 1, 0.9])
    ref = np.abs(Z).max(); ncol = max(nb, 3); c3 = [slice(0, ncol // 3), slice(ncol // 3, 2 * ncol // 3), slice(2 * ncol // 3, ncol)]
    span_tag = " (CR bands only)" if full else ""
    ax = fig.add_subplot(gs[0, c3[0]]); fp.map_db(ax, x, f, Z, ref=ref, colorbar=False); fp.mark_positions(ax, x[sel])
    ax.set_title(f"dense map ({len(x)} positions), N = {len(sel)} selected"); ax.set_xlabel("distance from clamp (µm)")
    ax = fig.add_subplot(gs[0, c3[1]])
    if show_arm in rec:
        zr = rec[show_arm]
        fp.map_db(ax, x, f, np.where(np.isnan(zr), 0, zr), ref=ref, colorbar=False); fp.mark_positions(ax, x[sel])
        ax.set_title(f"{show_arm}: held-out NRMSE {_nrmse(zr[h], Z[h], span):.1f} %{span_tag}")
    else:
        ax.set_title(f"{show_arm}: not computed"); ax.axis("off")
    ax.set_xlabel("distance from clamp (µm)")
    # per-position error, on the matched span
    ax = fig.add_subplot(gs[0, c3[2]])
    for arm, zr in rec.items():
        c, ls = ARM_STYLE[arm]
        err = [_nrmse(zr[i], Z[i], span) for i in range(len(x))]
        ax.plot(x, err, ls, color=c, lw=1, label=f"{arm} ({_nrmse(zr[h], Z[h], span):.1f} % held-out)")
    for xs in x[sel]: ax.axvline(xs, color="0.85", lw=0.6, zorder=0)
    ax.axvline(x[ipos], color="k", lw=0.8, ls="--")
    ax.set_yscale("log"); ax.set_ylim(1, 300); ax.set_xlabel("distance from clamp (µm)"); ax.set_ylabel("NRMSE at each position (%)")
    ax.legend(fontsize=8); ax.set_title(f"error along the lever{span_tag}")
    # spectrum at ipos (full width)
    axa = fig.add_subplot(gs[1, :]); axp = axa.twinx()
    ph = lambda z: np.where(span, np.degrees(np.angle(z)), np.nan)      # phase shown inside the CR bands only
    axa.plot(f / 1e3, 20 * np.log10(np.abs(Z[ipos]) + 1e-12), color="k", lw=1, label="measured")
    axp.plot(f / 1e3, ph(Z[ipos]), color="k", lw=0.5, alpha=0.3)
    for arm, zr in rec.items():
        c, ls = ARM_STYLE[arm]
        axa.plot(f / 1e3, 20 * np.log10(np.abs(zr[ipos]) + 1e-12), ls, color=c, lw=1, label=arm)
        axp.plot(f / 1e3, ph(zr[ipos]), ls, color=c, lw=0.5, alpha=0.5)
    if full:
        for b, m in R["masks"].items(): axa.axvspan(f[m].min() / 1e3, f[m].max() / 1e3, color="0.93", zorder=0); axa.text(f[m].mean() / 1e3, 0.97, b, transform=axa.get_xaxis_transform(), ha="center", va="top", fontsize=8, color="0.4")
    axa.set_xlabel("frequency (kHz)"); axa.set_ylabel("|Z| (dB V)"); axp.set_ylabel("phase (deg, faint)"); axp.set_ylim(-200, 200)
    tag = "measured position" if ipos in sel else "held out"
    axa.set_title(f"spectrum at x = {x[ipos]:.0f} µm ({tag})" + ("; EB arms are defined inside the CR bands" if full else "")); axa.legend(fontsize=8, loc="lower right", ncol=5)
    # mode shapes: one panel per CR band
    rows = []
    for j, b in enumerate(R["bands"]):
        m = R["masks"][b]; fbn, Zb = f[m], Z[:, m]
        ax = fig.add_subplot(gs[2, j] if full else gs[2, :])
        f0, _ = spectra.on_resonance_profile(fbn, Zb, (fbn.min(), fbn.max())); jf = int(np.argmin(np.abs(f - f0)))
        a0 = np.abs(Z[:, jf]); nrm = a0.max()
        ax.plot(x, a0 / nrm, color="k", lw=1.4, label="dense")
        nd0 = [n for n in spectra.nodes_from_profile(x, a0 / nrm) if x.min() + 5 <= n <= x.max() - 5]
        for n in nd0: ax.axvline(n, color="0.8", lw=0.8, ls=":")
        rows.append(f"{b} @ {f0/1e3:.0f} kHz  dense nodes {np.round(nd0, 1).tolist()}")
        for arm, zr in rec.items():
            c, ls = ARM_STYLE[arm]; a = np.abs(zr[:, jf])
            if np.all(np.isnan(a)): continue
            ax.plot(x, a / nrm, ls, color=c, lw=1.1, label=arm)
            nd = [n for n in spectra.nodes_from_profile(x, a / nrm) if x.min() + 5 <= n <= x.max() - 5]
            for n in nd: ax.plot(n, 0.02, "v", color=c, ms=5)
            err = [min(abs(n - q) for q in nd0) for n in nd] if nd0 and nd else []
            rows.append(f"    {arm:8s} nodes {np.round(nd, 1).tolist()}" + (f"  |err| {np.round(err, 1).tolist()} µm" if err else ""))
        ax.plot(x[sel], a0[sel] / nrm, "v", color="k", ms=6, label="measured")
        ax.set_ylim(0, 1.5); ax.set_xlabel("distance from clamp (µm)"); ax.set_title(f"{b}: mode shape at {f0/1e3:.0f} kHz", fontsize=9)
        if j == 0: ax.set_ylabel("|Z| on resonance (norm.)"); ax.legend(fontsize=7, ncol=2, loc="upper left")
    plt.show()
    print("\n".join(rows))
"""),
("md", "## Controls"),
("code", """
w_probe = W.Dropdown(options=["stiff", "soft"], value="stiff", description="probe")
w_band = W.Dropdown(options=["CR1", "CR2", "CR3", "full"], value="CR3", description="band")
w_N = W.IntSlider(3, 3, 30, description="N")
w_mode = W.Dropdown(options=["equispaced", "random", "custom"], value="equispaced", description="design")
w_seed = W.IntText(0, description="seed")
w_custom = W.Text("", description="custom x", placeholder="e.g. 50, 90, 130, 170 (µm from clamp)", layout=W.Layout(width="420px"))
w_rank = W.IntSlider(6, 1, 12, description="low-rank r")
w_arms = W.SelectMultiple(options=["GP", "low-rank", "EB", "EB+GP"], value=("GP", "low-rank", "EB+GP"), description="arms", rows=4)
w_show = W.Dropdown(options=["GP", "low-rank", "EB", "EB+GP"], value="GP", description="map of")
w_pos = W.IntSlider(80, 0, 160, description="spectrum @")
b_update = W.Button(description="Update", button_style="primary")
out = W.Output()

def _bands(*_):
    D = load(w_probe.value); w_band.options = list(D["bands"]) + ["full"]
    w_pos.max = len(D["x"]) - 1; w_pos.value = min(w_pos.value, w_pos.max)
w_probe.observe(_bands, "value"); _bands()

def refresh(run_eb=False):
    with out:
        clear_output(wait=True)
        R = compute(w_probe.value, w_band.value, w_N.value, w_mode.value, w_seed.value, w_custom.value, w_rank.value, w_arms.value, run_eb)
        w_pos.description = f"spectrum @ {R['x'][w_pos.value]:.0f} µm"
        draw(R, w_show.value, w_pos.value)
b_update.on_click(lambda _: refresh(False))
for w in (w_probe, w_band, w_N, w_mode, w_seed, w_rank, w_arms, w_show, w_pos):
    w.observe(lambda ch: refresh(False), "value")
display(W.VBox([W.HBox([w_probe, w_band, w_N, w_rank]), W.HBox([w_mode, w_seed, w_custom]),
                W.HBox([w_arms, w_show, w_pos, b_update]), out]))
refresh(False)
"""),
]
