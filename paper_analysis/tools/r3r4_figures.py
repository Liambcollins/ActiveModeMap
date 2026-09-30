"""Figures for the R3/R4 Multi75-G report. python tools/r3r4_figures.py <data root> [fig ...]"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import beimg
root = Path(sys.argv[1]); which = sys.argv[2:] or ["f1", "f2", "f3", "f4", "f5", "f6"]
R = HERE.parent / "results/r3r4"; F = HERE.parent / "figures/r3r4"; F.mkdir(parents=True, exist_ok=True)
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]; INK2 = "#52514e"
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5e0",
                     "grid.linewidth": 0.5, "axes.edgecolor": INK2, "lines.linewidth": 1.6, "legend.frameon": False, "figure.dpi": 150})
def tag(ax, s): ax.text(-0.14, 1.04, s, transform=ax.transAxes, fontweight="bold", fontsize=10, va="bottom")

if "f1" in which:   # R3 survey: d33, V_cpd, E and M along the lever, model vs measured
    S = json.load(open(R / "r3_survey.json")); t = pd.DataFrame(S["table"]); Mv = json.load(open(R / "r3_E_model_vs_survey.json"))
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2))
    ax[0].plot(t.x_um, t.d33_qs, "o-", color=C[0], label="as measured |P_QS|"); ax[0].plot(t.x_um, t.d33_qs_corr_no232, "s-", color=C[1], label="domain-CPD corrected")
    ax[0].axhline(S["cpd_split_no232"]["d33_uniform"], color=C[1], lw=0.8, ls=":"); ax[0].set(xlabel="stage x (µm)", ylabel="d33 QS (pm/V)", ylim=(0, 11), title="R3 survey, 350 nN"); ax[0].legend(); tag(ax[0], "a")
    ax[1].plot(t.x_um, t.vcpd_qs, "o-", color=C[0], label="QS band"); ax[1].plot(t.x_um, t.vcpd_cr, "s-", color=C[1], label="CR1")
    ax[1].set(xlabel="stage x (µm)", ylabel="V_cpd (V)", ylim=(3, 10), title="contact potential"); ax[1].legend(); ax[1].annotate("blind spot:\nV_cpd ill-defined", (232, 8.6), (185, 9.2), fontsize=7, color=INK2, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6)); tag(ax[1], "b")
    m = Mv["calfa_mu0.026"]
    ax[2].semilogy(t.x_um, t.E, "o", color=C[0], label="measured E"); ax[2].semilogy(t.x_um, t.E_corr_no232, "s", color=C[1], label="measured, CPD-corrected")
    ax[2].semilogy(m["x_eval"], m["E_eval"], "-", color=C[2], label="model (free air + survey)")
    ax[2].set(xlabel="stage x (µm)", ylabel="E = |P_CR1| / |P_QS|", title="in-situ enhancement"); ax[2].legend(); tag(ax[2], "c")
    for k, (lab, c) in enumerate((("calfa_muprior0.003", C[2]), ("calfa_mu0.026", C[3]), ("calfa_freemu", C[6]))):
        mm = Mv[lab]; x = np.array(mm["x"]); ax[3].plot(x[:-1], (np.array(mm["M_model"]) / np.array(mm["M_meas"]))[:-1], "o-", color=c, label=f"vs measured, {lab.split('_')[1]}")
    mm = Mv["calfa_mu0.026"]; ax[3].plot(x[:-1], (np.array(mm["M_model"]) / np.array(mm["M_corr"]))[:-1], "s--", color=C[1], label="vs CPD-corrected (mu 0.026)")
    ax[3].axhline(1, color=INK2, lw=0.6); ax[3].set(xlabel="stage x (µm)", ylabel="model M / measured M", ylim=(0.7, 1.4), title="model vs survey (M = E/Q)"); ax[3].legend(fontsize=6.5); tag(ax[3], "d")
    fig.tight_layout(); fig.savefig(F / "fig_r3_survey.png"); print("f1")

if "f2" in which:   # geometry: free air, dense map vs model, positions from InvOLS
    fa = json.load(open(R / "calfa_mu0.026.json"))["free_air"]; dv = json.load(open(R / "r3_dense_vs_model.json")); pos = pd.read_csv(R / "positions.csv")
    fig, ax = plt.subplots(1, 4, figsize=(13, 3.2)); x = np.array(fa["x_um"])
    for j in range(3):
        ax[0].plot(x, fa["shapes_meas"][j], "o", color=C[j], ms=4); ax[0].plot(x, fa["shapes_model"][j], "-", color=C[j], label=f"f{j+1} = {fa['f_meas_Hz'][j]/1e3:.1f} kHz")
    ax[0].set(xlabel="stage x (µm)", ylabel="|free mode slope| (norm.)", title="free-air walk (dots) vs fit"); ax[0].legend(); tag(ax[0], "a")
    xe = np.array(dv["x_equiv"]); xl = np.array(dv["x_label"]); mo = dv["models"]["calfa_mu0.026"]
    ax[1].plot(xe, dv["A1"], ".", color=C[0], ms=3, label="dense map CR1"); ax[1].plot(mo["x"], mo["m1"], "-", color=INK2, lw=1, label="model (no dense-map input)")
    ax[1].set(xlabel="InvOLS-equivalent x (µm)", ylabel="|Z| at CR1 peak (V)", title=f"CR1 profile, NRMSE {100*mo['nrmse_CR1_profile']:.1f} %"); ax[1].legend(); tag(ax[1], "b")
    ax[2].plot(xe, dv["A2"], ".", color=C[1], ms=3, label="dense map CR2"); ax[2].plot(mo["x"], mo["m2"], "-", color=INK2, lw=1, label="model")
    ax[2].axvline(dv["summary"]["dense_CR2_node_equiv"], color=C[1], lw=0.6, ls=":"); ax[2].set(xlabel="InvOLS-equivalent x (µm)", ylabel="|Z| at CR2 peak (V)", title=f"CR2 node {dv['summary']['dense_CR2_node_equiv']:.0f} µm (model {mo['CR2_node_model']:.0f})"); ax[2].legend(); tag(ax[2], "c")
    for k, (st, c) in enumerate((("preflight", C[0]), ("survey", C[1]), ("imaging (16:13)", C[3]), ("held-out walk", C[2]), ("dense map", C[6]))):
        q = pos[pos.stage == st]; ax[3].plot(q.x_label, q.offset_um, "o" if st != "dense map" else ".", color=c, ms=4 if st != "dense map" else 2, label=st.replace(" (16:13)", ""))
    ax[3].axhline(0, color=INK2, lw=0.6); ax[3].set(xlabel="stage x label (µm)", ylabel="x from InvOLS − label (µm)", title="where the laser really was (R3)"); ax[3].legend(fontsize=6.5); tag(ax[3], "d")
    fig.tight_layout(); fig.savefig(F / "fig_r3_geometry.png"); print("f2")

if "f3" in which:   # held-out walk
    H = json.load(open(R / "r3_heldout.json")); Hb = json.load(open(R / "r3_heldout_bias.json"))
    fig, ax = plt.subplots(1, 2, figsize=(8, 3.0))
    names = {"a_labels": "positions as labelled", "b_invols_x": "+ InvOLS positions", "c_invols_x_fshift": "+ CR shift removed"}
    for k, (case, lab) in enumerate(names.items()):
        rr = pd.DataFrame(H["cases"][case]["rows"]); ax[0].plot(rr["rank"], rr.nrmse_CR1, "o-", color=C[k], label=lab)
    b = Hb["best"][0]; ax[0].plot([b["rank"]], [b["nrmse_CR1"]], "*", ms=10, color=C[3], label=f"+ survey at V = {b['V_eff']:+.1f} V")
    ax[0].set(xlabel="rank", ylabel="CR1 NRMSE on held-out walk (%)", ylim=(0, 170), title="held-out walk (7 midpoints, 3 h later)"); ax[0].legend(fontsize=6.5); tag(ax[0], "a")
    sc = pd.DataFrame(Hb["scan"]); u = sc[(sc["rank"] == 5) & (sc.spot == 1)]
    ax[1].plot(u.V_eff, u.nrmse_CR1, "-", color=C[0], label="CR1"); ax[1].plot(u.V_eff, u.nrmse_CR2, "-", color=C[1], label="CR2")
    ax[1].set(xlabel="survey bias used as the prediction (V)", ylabel="NRMSE (%)", title="walk at 0 V looks like survey at +5 V"); ax[1].legend(); tag(ax[1], "b")
    fig.tight_layout(); fig.savefig(F / "fig_r3_heldout.png"); print("f3")

def be_map(ax, path, M, inv, title, vmax=12, lim=None):
    b = beimg.load_be(path); d = b["A"] / M * inv / 32 * 1e12
    im = ax.imshow(d, cmap="Blues", vmin=0, vmax=vmax, origin="lower"); ax.set(title=title, xticks=[], yticks=[]); ax.grid(False); return im
def qs_map(ax, path, title, vmax=12, scale=1.0):
    q = beimg.load_qs(path); d = q["amp_m"] * scale * 1e12 / q["drive_V"]
    im = ax.imshow(d, cmap="Blues", vmin=0, vmax=vmax, origin="lower", extent=[0, q["scan_size"] * 1e6] * 2); ax.set(title=title); ax.grid(False); return im

if "f4" in which:   # R3 frames
    cal = json.load(open(R / "r3_frame_calibration.json")); M = cal["states"]["survey_E"]["M"]; inv = cal["invols_img"]; d = root / "DomainsB_SCMPIT_R3"
    fig, ax = plt.subplots(1, 6, figsize=(15, 2.9), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 1, 1.5]))
    for k, (n, lab) in enumerate((("A_be_0V", "BE 0 V"), ("A_be_Vhalf", "BE +2.5 V"), ("A_be_Vcpd", "BE +5.1 V (V_cpd)"), ("A_be_0V_repeat", "BE 0 V repeat"))):
        im = be_map(ax[k], d / f"52_image_be/BE_{n}.hf5", M, inv, lab)
    qs_map(ax[4], d / "QImg0079.ibw", "QS 20 kHz, 0 V"); ax[4].set(xlabel="µm")
    fig.colorbar(im, ax=ax[:5], shrink=0.8, label="|d33| per pixel (pm/V)")
    t = pd.read_csv(R / "r3_frames.csv"); be = t[t.kind == "BE"]
    for k, st in enumerate(("survey_E", "model_E", "mixed_E", "agent_500_267")):
        q = be[be.state == st]; ax[5].plot(range(4), q.d33_best, "o-", color=C[k], label=f"{st} (M {q.M.iloc[0]:.2f})")
    qsv = t[(t.kind == "QS") & t.frame.str.startswith("A_qs")].d33_best.iloc[0]
    ax[5].axhline(qsv, color=INK2, ls="--", lw=0.8); ax[5].text(3.05, qsv, "QS contrast", fontsize=6.5, va="bottom", color=INK2)
    ax[5].axhline(7.0, color=C[1], ls=":", lw=0.8); ax[5].text(3.05, 7.0, "survey, corrected", fontsize=6.5, va="top", color=C[1])
    ax[5].set(xticks=range(4), xticklabels=["0 V", "+2.5", "+5.1", "0 V rep"], ylabel="two-domain |d33| (pm/V)", ylim=(3, 9), title="R3 BE: calibration choices"); ax[5].legend(fontsize=6)
    ax[5].grid(True)
    fig.savefig(F / "fig_r3_frames.png", bbox_inches="tight"); print("f4")

if "f5" in which:   # R4 frames
    d = root / "DomainsB_SCMPIT_R4"; cal = json.load(open(R / "r4_frame_calibration.json")); M = cal["states"]["M350"]["M"]
    fig, ax = plt.subplots(1, 6, figsize=(15, 2.9))
    for k, (n, inv, lab) in enumerate((("be_Vcpd", 826.4e-9, "BE ±1, 350 nN, V_cpd"), ("be_0V_700nN", 846.5e-9, "BE ±1, 700 nN, 0 V"),
                                        ("bew4_0V_700nN_128px", 842e-9, "BE ±4, 700 nN, 0 V"), ("bew4_Vcpd_700nN_128px", 835.1e-9, "BE ±4, 700 nN, V_cpd"))):
        im = be_map(ax[k], d / f"52_image_be/BE_{n}.hf5", M, inv, lab)
    qs_map(ax[4], d / "QImg0500.ibw", "QS 6 µm, 350 nN"); qs_map(ax[5], d / "QImg0505.ibw", "QS 6 µm, 700 nN")
    fig.colorbar(im, ax=ax, shrink=0.8, label="|d33| per pixel (pm/V)"); fig.savefig(F / "fig_r4_frames.png", bbox_inches="tight"); print("f5")

if "f6" in which:   # R4 summary: BE vs QS by load; Q-vs-A
    t = pd.read_csv(R / "r4_frames.csv"); fig, ax = plt.subplots(1, 3, figsize=(12, 3.2))
    be = t[(t.kind == "BE") & (t.state == "M350")].reset_index(drop=True); qs = t[t.kind == "QS"].reset_index(drop=True)
    for L, c in ((350, C[0]), (700, C[1])):
        b = be[be.load_nN == L]; q = qs[qs.load_nN == L]
        ax[0].plot(np.flatnonzero(be.load_nN == L), b.d33_best, "o", color=c, label=f"BE {L} nN (M 350 nN)")
    bc = t[(t.kind == "BE") & (t.state == "M700_cond")]
    ax[0].plot([be.index[be.frame == f][0] for f in bc.frame], bc.d33_best, "x", color=C[1], label="BE 700 nN (M 700, conditional)")
    for L, c, ls in ((350, C[0], "--"), (700, C[1], ":")):
        v = qs[(qs.load_nN == L) & qs.frame.str.contains("6um")].d33_best.mean(); ax[0].axhline(v, color=c, ls=ls, lw=0.8); ax[0].text(len(be) - 0.5, v, f"QS 6 µm {L} nN", fontsize=6.5, color=c, ha="right", va="bottom")
    ax[0].set(xticks=range(len(be)), ylabel="|d33| (pm/V)", ylim=(5, 11), title="R4 BE frames vs QS"); ax[0].set_xticklabels(be.frame, rotation=70, fontsize=5.5); ax[0].legend(fontsize=6); tag(ax[0], "a")
    for k, (pat, c) in enumerate((("blind-spot", C[0]), ("6um", C[1]), ("12um", C[2]), ("unlogged", C[3]))):
        q = qs[qs.frame.str.contains(pat)]; ax[1].plot(q.load_nN, q.d33_best, "o", color=c, label=pat)
    ax[1].set(xlabel="load (nN)", ylabel="QS two-domain |d33| (pm/V)", title="QS d33 vs load", ylim=(5, 11)); ax[1].legend(fontsize=6.5); tag(ax[1], "b")
    b = beimg.load_be(root / "DomainsB_SCMPIT_R4/52_image_be/BE_be_Vcpd.hf5"); A = b["A"].ravel(); Q = b["Q"].ravel(); m = (Q > 100) & (Q < 600)
    ax[2].hexbin(Q[m], (A * Q)[m] / np.median((A * Q)[m]), gridsize=50, cmap="Blues", mincnt=1, bins="log")
    qq = np.linspace(Q[m].min(), Q[m].max(), 10); ax[2].plot(qq, qq / np.median(Q[m]), color=C[1], lw=1, label="peak ∝ Q (A constant)"); ax[2].axhline(1, color=INK2, lw=0.8, ls="--", label="peak constant")
    ax[2].set(xlabel="per-pixel Q", ylabel="peak = A·Q (norm.)", ylim=(0.5, 1.5), title="be_Vcpd 350 nN: per-pixel Q scatter"); ax[2].legend(fontsize=6.5); ax[2].grid(False); tag(ax[2], "c")
    fig.tight_layout(); fig.savefig(F / "fig_r4_summary.png"); print("f6")
