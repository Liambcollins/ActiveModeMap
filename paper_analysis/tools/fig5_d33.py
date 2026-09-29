"""Fig. 5: quantitative d33 on the stiff probe across two domains, with the enhancement E(x) taken
from the fast mode map instead of a dense calibration.

(a) two-domain decomposition at one position: Z on each domain, P = (Z1 - Z2)/2, and E(x) from a
    single domain at 0 V vs the two-domain E (the electrostatic channel);
(b) E at never-visited positions predicted from the two-domain P map (leave-one-out) vs measured;
(c) d33 along the lever: quasi-static, CR1 / E_FMM, CR1 / blind EB (raw conventions);
(d, e) images at positions A and B: CR1 amplitude, quasi-static d33, and CR1 / E_FMM(x) d33;
(f) d33 per domain from the images against spectroscopy.

    python tools/fig5_d33.py
"""
from __future__ import annotations

import datetime as dt, json, sys
from pathlib import Path

import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import fmmpaper as F  # noqa: E402
from fmmpaper import axis, calib, domains, ebfit, eb_fit, images, physrec, spectra, plotting as fp  # noqa: E402

L_UM, VAC = 226.5, 1.0
CAMP, D0 = "DomainsB_SCMPIT_R2", dt.datetime(2026, 9, 20, 22)
X0_DENSE = 28.1


def survey_table():
    s = F.load("scmpitB_r2_bias"); tl = calib.timeline(CAMP, D0); an = calib.frame_anchors(CAMP, tl, L_UM)
    x0 = float(an.set_index("stage").loc["10_bias_survey", "x0"])
    pre = axis.invols_interpolator(F.load("scmpitB_r2_preflight"))
    inv_fn = lambda x: float(np.atleast_1d(pre(x))[0]) * float(calib.factor_at(tl, "10_bias_survey", x)[0])
    t = domains.transfer_table(s, inv_fn, VAC); t["x_lever_um"] = t.x_um - x0
    return s, t, x0, tl, an


def e_from_pmap(s, x_stage, cal, loo=True):
    """E at the survey positions from EB fits of the two-domain P map (CR1 band + quasi-static band).
    loo: each position predicted from the other seven."""
    Pc = domains.decompose_series(s, band="CR1"); Pq = domains.decompose_series(s, band="QS"); qs = s.probe.qs_band_Hz
    zc, fc = physrec.band_slice(Pc["freq_Hz"], Pc["P"], s.probe.bands_Hz["CR1"], n_max=250)
    zq, fq = physrec.band_slice(Pq["freq_Hz"], Pq["P"], tuple(qs), n_max=100)
    n = len(x_stage); E_true = np.abs(zc).max(1) / np.abs(zq).mean(1); E_eb = np.zeros(n); E_gp = np.zeros(n)
    fits_all = None
    for j in range(n):
        keep = np.array([i for i in range(n) if i != j]) if loo else np.arange(n)
        oc = ebfit.rec_eb2(x_stage, keep, zc[keep], fc, cal, "CR1", x_eval=x_stage[[j]])
        cq = json.loads(json.dumps(cal)); cq["P"]["log_alpha"] = oc["theta"]["log_alpha"]
        oq = ebfit.rec_eb2(x_stage, keep, zq[keep], fq, cq, "QS", free=("eps",), x_eval=x_stage[[j]])
        E_eb[j] = np.abs(oc["Zeb"][0]).max() / np.abs(oq["Zeb"][0]).mean(); E_gp[j] = np.abs(oc["Zrec"][0]).max() / np.abs(oq["Zrec"][0]).mean()
    return E_true, E_eb, E_gp, (zc, fc, zq, fq)


def e_at(s, x_stage, cal, x_query):
    """E at arbitrary stage positions from the full 8-position two-domain P map (no hold-out)."""
    Pc = domains.decompose_series(s, band="CR1"); Pq = domains.decompose_series(s, band="QS"); qs = s.probe.qs_band_Hz
    zc, fc = physrec.band_slice(Pc["freq_Hz"], Pc["P"], s.probe.bands_Hz["CR1"], n_max=250)
    zq, fq = physrec.band_slice(Pq["freq_Hz"], Pq["P"], tuple(qs), n_max=100)
    sel = np.arange(len(x_stage)); xq = np.asarray(x_query, float)
    oc = ebfit.rec_eb2(x_stage, sel, zc, fc, cal, "CR1", x_eval=xq)
    cq = json.loads(json.dumps(cal)); cq["P"]["log_alpha"] = oc["theta"]["log_alpha"]
    oq = ebfit.rec_eb2(x_stage, sel, zq, fq, cq, "QS", free=("eps",), x_eval=xq)
    return np.abs(oc["Zrec"]).max(1) / np.abs(oq["Zrec"]).mean(1), np.abs(oc["Zeb"]).max(1) / np.abs(oq["Zeb"]).mean(1)


def main():
    fp.setup()
    cal = ebfit.load_calibration("stiff", "mass")
    s, t, x0_s, tl, an = survey_table()
    xs_stage = t.x_lever_um.values + x0_s
    # frame for the survey: c from the R2 live-capture fit shifted by the ruler drift (live_ebgp_d33)
    live = json.load(open(F.config.RESULTS_DIR / "live_ebgp_d33_stiff_r2.json"))
    c_s = json.loads(json.dumps(cal)); c_s["P"]["c_um"] = live["c_fitted"] + (x0_s - live["x0"]["design"])
    E_true, E_eb, E_gp, (zc, fc, zq, fq) = e_from_pmap(s, xs_stage, c_s)
    # blind EB (Fig. 6 central) for reference
    c6 = json.load(open(F.config.RESULTS_DIR / "_cache_fig6.json"))["R2"]["central"]
    st = eb_fit.ContactState("R2", np.zeros(3), 0.0, np.array([]), np.array([]), L_um=L_UM)
    E_blind = eb_fit.enhancement(np.array(c6["p"]), st, t.x_lever_um.values)[0]
    conv = t.invols.values / s.probe.amp_divisor / VAC * 1e12
    d33_qs = t.d33_qs.values; d33_fmm = t.P_cr.values * conv / E_gp; d33_fmm_eb = t.P_cr.values * conv / E_eb; d33_blind = t.P_cr.values * conv / E_blind
    # single-domain E at 0 V for panel a
    qm = s.band("QS"); band = s.probe.bands_Hz["CR1"]; E1 = []; E2 = []
    for j in range(len(xs_stage)):
        for spot, dest in ((1, E1), (2, E2)):
            i = int(np.intersect1d(s.where(bias_V=0), s.where(spot=spot))[0]); pk = spectra.peak(s.freq_Hz, s.Z[i, j], band)
            dest.append(pk["amp"] / np.abs(s.Z[i, j, qm].mean()))
    # ---------------------------------------------------------------- images at A and B
    root = F.config.data_root() / "DomainsB_SCMPIT_R2/50_quant_imaging"
    J = json.loads((root / "image_cr1_quant.json").read_text()); ims = {im["name"]: im for im in J["images"]}
    x0i = float(calib.x0_at(an, np.datetime64(J["started"].replace(" ", "T")))[0])
    cs = domains.cpd_split(t, mask=np.ones(len(t), bool))
    xl = t.x_lever_um.values; cplx = lambda a, b: t[a].values + 1j * t[b].values
    rq = cplx("bq_re", "bq_im") / cplx("Pq_re", "Pq_im"); rc = cplx("bc_re", "bc_im") / cplx("Pc_re", "Pc_im")
    ri = lambda r, x: np.interp(x, xl, r.real) + 1j * np.interp(x, xl, r.imag)
    delta = cs["delta_V"]; Vbar = float(J["v_cpd_V"]); V1, V2 = Vbar - delta, Vbar + delta
    IM = {}; rows = []
    for pos, xs in (("A", 154.9), ("B", 232.0)):
        X = xs - x0i
        E_img_fmm, E_img_eb = e_at(s, xs_stage, c_s, [X + x0_s])                      # same lever position, survey frame
        rd = lambda n: images.read(root / ims[n]["path"].split(chr(92))[-1], ims[n]["drive_V"])
        zq_i, zc0 = rd(f"{pos}_qs20k_0V"), rd(f"{pos}_cr1_0V")
        vname = next(n for n in ims if n.startswith(f"{pos}_cr1_V") and abs(ims[n]["bias_V"] - Vbar) < 1e-6); zcv = rd(vname)
        m1, m2 = images.domain_masks(zc0)
        pq = images.cpd_correct(zq_i, m1, m2, 0.0, V1, V2, ri(rq, X), delta)
        pc = images.cpd_correct(zcv, m1, m2, Vbar, V1, V2, ri(rc, X), delta)
        e1, e2 = images.interior(m1), images.interior(m2)
        d_q = np.abs(pq); d_c = np.abs(pc) / float(E_img_fmm[0])
        E_img = abs(images.domain_contrast(zc0, m1, m2)) / abs(images.domain_contrast(zq_i, m1, m2))
        IM[pos] = dict(cr=np.abs(zc0), qs=d_q, d33=d_c, m1=m1, m2=m2, X=X, E_fmm=float(E_img_fmm[0]))
        rows.append(dict(pos=pos, lever_um=X, E_fmm=float(E_img_fmm[0]), E_fmm_eb=float(E_img_eb[0]), E_image_contrast=E_img,
                         qs_d1=np.median(d_q[e1]), qs_d2=np.median(d_q[e2]), fmm_d1=np.median(d_c[e1]), fmm_d2=np.median(d_c[e2]),
                         d33_spectroscopy=cs["d33_uniform"]))
    imgtab = pd.DataFrame(rows); print(imgtab.round(2).to_string(index=False))
    # ---------------------------------------------------------------- figure
    W = fp.WIDTH_IN["double"]; fig = plt.figure(figsize=(W, W * 0.66))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 0.75], hspace=0.4)
    g1 = gs[0].subgridspec(1, 3, wspace=0.4); axa, axb, axc = (fig.add_subplot(g1[0, i]) for i in range(3))
    xlv = t.x_lever_um.values
    # (a) single-domain vs two-domain E
    axa.semilogy(xlv, E1, "^", color=fp.DOMAIN_C[1], ms=3.2, mec="none", label="domain 1, 0 V")
    axa.semilogy(xlv, E2, "v", color=fp.DOMAIN_C[2], ms=3.2, mec="none", label="domain 2, 0 V")
    axa.semilogy(xlv, E_true, "o-", color="k", ms=3.2, lw=0.8, mfc="w", label="two-domain P")
    axa.set_xlabel("distance from clamp x (µm)"); axa.set_ylabel("CR1 enhancement |Z$_{CR1}$| / |Z$_{QS}$|")
    axa.legend(fontsize=5.3, loc="lower left"); axa.set_title("single domain at 0 V vs two-domain P", fontsize=6.5)
    # (b) predicted vs measured E (LOO)
    axb.semilogy(xlv, E_true, "o", color="k", ms=3.4, mfc="w", label="measured (two-domain)")
    axb.semilogy(xlv, E_gp, "s", color=fp.C["ebgp"], ms=3, mec="none", label="predicted from the other 7 positions")
    axb.semilogy(xlv, E_blind, "--", color="0.5", lw=0.8, label="blind EB (no amplitudes)")
    axb.set_xlabel("distance from clamp x (µm)"); axb.set_ylabel("E at a never-visited position"); axb.legend(fontsize=5.3, loc="lower left")
    r = E_true / E_gp; axb.set_title(f"E predicted to {r.min():.2f}–{r.max():.2f} of measured", fontsize=6.5)
    # (c) d33 routes
    axc.plot(xlv, d33_qs, "o-", color="k", ms=3, lw=0.8, mfc="w", label="quasi-static")
    axc.plot(xlv, d33_fmm, "s-", color=fp.C["ebgp"], ms=3, lw=0.8, mec="none", label="CR1 ÷ E from the mode map")
    axc.plot(xlv, d33_blind, "d--", color="0.5", ms=2.6, lw=0.7, mec="none", label="CR1 ÷ blind EB")
    axc.set_xlabel("distance from clamp x (µm)"); axc.set_ylabel("d$_{33}$ (pm/V)"); axc.set_ylim(0, 13); axc.legend(fontsize=5.3, loc="lower left")
    axc.set_title("d$_{33}$ along the lever", fontsize=6.5)
    # (d, e) images; (f) summary
    g2 = gs[1].subgridspec(1, 8, wspace=0.08, width_ratios=[1, 1, 1, 1, 1, 0.06, 0.9, 1.4])
    vmax = 12; panels = [("A", "cr", "A: CR1 amplitude, 0 V", "magma", None), ("A", "qs", "A: quasi-static d$_{33}$", "viridis", vmax), ("A", "d33", "A: CR1 ÷ E$_{FMM}$", "viridis", vmax),
                         ("B", "qs", "B: quasi-static d$_{33}$", "viridis", vmax), ("B", "d33", "B: CR1 ÷ E$_{FMM}$", "viridis", vmax)]
    for k, (pos, key, title, cmap, vm) in enumerate(panels):
        ax = fig.add_subplot(g2[0, k]); Z = IM[pos][key]
        h = ax.imshow(Z, cmap=cmap, vmin=0, vmax=vm if vm else np.percentile(Z, 99.5), origin="lower", aspect="equal")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title, fontsize=6, pad=3)
        if key != "cr": cbh = h
        if k == 0: fp.panel_label(ax, "d", dx=-0.1, dy=1.12)
        if k == 3: fp.panel_label(ax, "e", dx=-0.1, dy=1.12)
        if key == "d33":
            e1, e2 = images.interior(IM[pos]["m1"]), images.interior(IM[pos]["m2"])
            ax.text(0.03, 0.95, f"{np.median(Z[e1]):.1f}", transform=ax.transAxes, color="w", fontsize=6, va="top"); ax.text(0.03, 0.05, f"{np.median(Z[e2]):.1f}", transform=ax.transAxes, color="w", fontsize=6, va="bottom")
        if key == "qs":
            e1, e2 = images.interior(IM[pos]["m1"]), images.interior(IM[pos]["m2"])
            ax.text(0.03, 0.95, f"{np.median(Z[e1]):.1f}", transform=ax.transAxes, color="w", fontsize=6, va="top"); ax.text(0.03, 0.05, f"{np.median(Z[e2]):.1f}", transform=ax.transAxes, color="w", fontsize=6, va="bottom")
    cax = fig.add_subplot(g2[0, 5]); cb = fig.colorbar(cbh, cax=cax); cb.set_label("d$_{33}$ (pm/V)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    axf = fig.add_subplot(g2[0, 7]); yy = np.arange(len(imgtab))
    axf.axvline(cs["d33_uniform"], color="k", lw=0.8, ls="--", label=f"spectroscopy, {cs['d33_uniform']:.2f}")
    for j, (col, lab, c, mk, mfc) in enumerate((("qs_d1", "QS image, domain 1", fp.DOMAIN_C[1], "o", "w"), ("qs_d2", "QS image, domain 2", fp.DOMAIN_C[2], "o", "w"),
                                              ("fmm_d1", "CR1 ÷ E$_{FMM}$, domain 1", fp.DOMAIN_C[1], "s", fp.DOMAIN_C[1]), ("fmm_d2", "CR1 ÷ E$_{FMM}$, domain 2", fp.DOMAIN_C[2], "s", fp.DOMAIN_C[2]))):
        axf.plot(imgtab[col], yy + (j - 1.5) * 0.14, mk, color=c, ms=3.4, mfc=mfc, label=lab)
    axf.set_yticks(yy); axf.set_yticklabels([f"{p} ({x:.0f} µm)" for p, x in zip(imgtab.pos, imgtab.lever_um)], fontsize=6); axf.set_ylim(-0.6, 1.6); axf.set_xlim(0, 12)
    axf.set_xlabel("d$_{33}$ from the image (pm/V)"); axf.legend(fontsize=4.8, loc="upper center", bbox_to_anchor=(0.5, -0.32), ncol=2, handlelength=1.2, frameon=False, columnspacing=0.8); fp.panel_label(axf, "f", dx=-0.45, dy=1.12)
    axf.set_title("image d$_{33}$ vs spectroscopy", fontsize=6.5)
    for ax, lab in ((axa, "a"), (axb, "b"), (axc, "c")):
        fp.panel_label(ax, lab, dx=-0.25)
    path = F.config.FIG_DIR / "Fig5_d33.png"; fig.savefig(path, dpi=600); fig.savefig(str(path).replace(".png", ".pdf")); plt.close(fig); print("wrote", path)
    json.dump(dict(survey=dict(x_lever_um=xlv.tolist(), E_two_domain=E_true.tolist(), E_loo_ebgp=E_gp.tolist(), E_loo_eb=E_eb.tolist(), E_blind=E_blind.tolist(),
                               E_single_0V_d1=E1, E_single_0V_d2=E2, d33_qs=d33_qs.tolist(), d33_cr_over_Efmm=d33_fmm.tolist(), d33_cr_over_blind=d33_blind.tolist()),
                   images=imgtab.to_dict("records"), cpd=dict(delta_V=delta, d33_uniform=cs["d33_uniform"], Vbar=Vbar)),
              open(F.config.RESULTS_DIR / "fig5_d33.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
