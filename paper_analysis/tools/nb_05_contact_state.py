from nbbuild import SETUP

CELLS = [
("md", """
# 05 · Laser-frame drift, node positions and the contact state (Phase 2d, feeds Fig. 5)

The R2 report states that every node moved 11–14 µm toward the free end between Run 1 and Run 2,
and reads this as a change of the contact boundary condition. That comparison puts both dense
maps in one fixed frame (clamp at stage 6.1 µm).

**The frame is not fixed.** InvOLS(x) of a tip-loaded cantilever follows
1/[(x − x₀)²(3L − (x − x₀))] whatever the contact stiffness. Fitting that shape to every short
walk gives x₀, where the clamp really sits in stage coordinates at that time. This notebook:

1. measures x₀(t) through both campaigns (`calib.frame_anchors`);
2. puts the dense-map nodes into the lever frame, using x₀ at the time each position was measured;
3. refits the blind Euler–Bernoulli model for each contact state (frequencies, Q and static shape
   only) and asks whether it predicts the node positions it never saw.
"""),
("code", SETUP),
("code", """
import datetime as dt
from fmmpaper import calib, eb_fit
CAMP = {"R1": ("DomainsB_SCMPIT", dt.datetime(2026, 9, 20, 11), "30_dense_grid"),
        "R2": ("DomainsB_SCMPIT_R2", dt.datetime(2026, 9, 20, 22), "75_dense_grid")}
L_UM = 226.5          # lever length from the R1 pre-flight static-shape fit (0.8 % rms)
tl, an = {}, {}
for tag, (camp, d0, _) in CAMP.items():
    tl[tag] = calib.timeline(camp, d0)
    an[tag] = calib.frame_anchors(camp, tl[tag], L_UM)
pd.concat(an).round(3)
"""),
("code", """
fig, axs = fp.figure("double", aspect=0.3, ncols=2)
for tag, a in an.items():
    c = "0.55" if tag == "R1" else "k"
    axs[0].plot(a.t, a.x0, "o-", color=c, ms=4, label=tag)
    axs[1].plot(a.t, a.gain_rel, "o-", color=c, ms=4, label=tag)
axs[0].set_ylabel("x₀ in stage frame (µm)")
axs[1].set_ylabel("InvOLS gain / pre-flight gain"); axs[1].set_ylim(0.9, 1.1)
for ax in axs:
    ax.tick_params(axis="x", rotation=30); ax.legend()
fig.tight_layout(); fp.save(fig, "05_frame_drift")
"""),
("md", """
**Reading.**
- The detector gain stays within ±2–5 % all night.
- The clamp position drifts in stage coordinates. R1: 5.7 → 8.7 µm in the first 1.5 h, then
  21.2 µm at the close-out (after the load ladder). R2 starts at 23.8 µm, continuous with R1's
  close-out. It drifts ~0.9 µm/h overnight (25.0 → 29.8 µm) and reaches 39.3 µm after the
  morning load extension.
- So most of what looked like InvOLS "ageing" (SI-a) is the laser moving along the lever, by
  about 1 µm/h plus jumps during load excursions. The per-stop InvOLS is still the right number
  for pm/V conversion. Positions, however, have to be put into the lever frame.
- The R2 "free end" at stage 232 µm is really ~208 µm from the clamp (~18 µm inside the tip).
  This is why the R2 free end looked much less node-like than in R1.
"""),
("md", "## Dense-map nodes in the lever frame"),
("code", """
states, node_rows = {}, []
for tag, (camp, d0, dense) in CAMP.items():
    s = F.load(f"scmpitB_{tag.lower()}_dense")
    t_at_x = tl[tag][tl[tag].stage == dense].groupby("x_um").t.median()
    fr, nodes = {}, {}
    for b in ("CR1", "CR2", "CR3"):
        pk = spectra.peaks_along_x(s.freq_Hz, s.Z[0], s.probe.bands_Hz[b])
        fr[b] = float(np.median(pk["f_Hz"]))
        if b == "CR1":
            Q1 = float(np.nanmedian(pk["Q"])); continue
        lever = []
        for n in spectra.nodes_from_profile(s.x_um, pk["amp"]):
            tn = t_at_x.iloc[int(np.argmin(np.abs(t_at_x.index.values - n)))]
            x0n = float(calib.x0_at(an[tag], tn)[0])
            lever.append(n - x0n)
            node_rows.append(dict(run=tag, mode=b, stage_um=n, old_frame_um=n - 6.1,
                                  x0_at_time=x0n, lever_frame_um=n - x0n, time=tn))
        nodes[b] = lever
    sv = tl[tag][(tl[tag].stage == "10_bias_survey") & tl[tag].ratio.between(0.7, 1.6)].groupby("x_um").invols.median()
    x0s = float(an[tag].set_index("stage").loc["10_bias_survey", "x0"])
    xs = sv.index.values - x0s; keep = xs < L_UM - 10
    states[tag] = eb_fit.ContactState(f"{tag} dense-map state", np.array([fr["CR1"], fr["CR2"], fr["CR3"]]),
                                      Q1, xs[keep], sv.values[keep], nodes, L_um=L_UM)
nodes_tbl = pd.DataFrame(node_rows)
nodes_tbl.round(1)
"""),
("code", """
piv = nodes_tbl.assign(k=nodes_tbl.groupby(["run", "mode"]).cumcount()).pivot_table(
    index=["mode", "k"], columns="run", values=["old_frame_um", "lever_frame_um"])
piv[("shift", "old frame")] = piv[("old_frame_um", "R2")] - piv[("old_frame_um", "R1")]
piv[("shift", "lever frame")] = piv[("lever_frame_um", "R2")] - piv[("lever_frame_um", "R1")]
piv.round(1)
"""),
("md", """
In the fixed frame the nodes appear to move +11 to +14 µm. In the lever frame they move −1 to −6 µm,
toward the clamp. For R1 the dense map was taken between the last early anchor (13:52) and the
close-out (22:29), with the load ladder in between. The linear x₀ interpolation used for R1
therefore carries about ±1.5 µm of extra uncertainty.
"""),
("md", """
## Blind Euler–Bernoulli fit of each contact state

The fit sees CR1–CR3, Q₁ and the static InvOLS shape (lever frame, L = 226.5 µm); nodes are
withheld (`use_nodes=False`). A second fit adds the nodes to check consistency. Expect ~3–5 min per
fit on the laptop.
"""),
("code", """
fits = {}
for tag, st in states.items():
    for use_nodes in (False, True):
        r = eb_fit.fit(st, use_nodes=use_nodes)
        s = eb_fit.summarize(r.x, st); s["use_nodes"] = use_nodes; s["p"] = r.x.tolist()
        fits[(tag, use_nodes)] = s
rows = []
for (tag, un), s in fits.items():
    rows.append(dict(run=tag, fit="with nodes" if un else "blind", k_star_over_k=s["k_ratio"],
                     kcone_over_k=s["kcone_ratio"], setback_um=s["setback_um"], zeta=s["zeta"],
                     tip_height_um=s["tip_height_um"], cr_err_pct=s["cr_err_pct"], Q_model=s["Q_model"],
                     static_rms_pct=s["static_rms_pct"]))
pd.DataFrame(rows).round(3)
"""),
("code", """
cmp = []
for tag in states:
    blind = fits[(tag, False)]["nodes_model_um"]
    for mode in ("CR2", "CR3"):
        for i, meas in enumerate(states[tag].nodes_um.get(mode, [])):
            pred = min(blind[mode], key=lambda v: abs(v - meas)) if blind[mode] else np.nan
            cmp.append(dict(run=tag, mode=mode, measured_lever_um=meas, blind_EB_um=pred, diff_um=pred - meas))
cmp = pd.DataFrame(cmp); cmp.round(2)
"""),
("code", """
fig, ax = fp.figure("single", aspect=0.8)
for tag, mk in (("R1", "o"), ("R2", "s")):
    d = cmp[cmp.run == tag]
    ax.plot(d.measured_lever_um, d.blind_EB_um, mk, color="k" if tag == "R2" else "0.55", ms=5, label=tag)
lim = [80, 175]; ax.plot(lim, lim, color="0.8", lw=0.8)
old = nodes_tbl.groupby(["run", "mode", "stage_um"]).first().reset_index()
ax.set_xlabel("measured node, lever frame (µm from clamp)"); ax.set_ylabel("blind EB prediction (µm)")
ax.legend(); ax.set_xlim(lim); ax.set_ylim(lim)
fp.save(fig, "05_nodes_blind_prediction")
"""),
("md", """
**Result.**
- Blind fits that never saw a node place all six interior nodes within ~0–3 µm of the
  drift-corrected measurements. Adding the nodes to the fit barely changes the parameters.
- Between the two dense maps the contact became stiffer (k*/k_lever ~480 → ~750) and the
  effective tip setback grew by ~3 µm (≈2.5 → 5 µm). Frequencies rose 3–8 % and the nodes moved a
  few µm toward the clamp. This is consistent with a blunter tip after the R1 load ladder.

**What changes in the manuscript.**
- "Every node moved 11–14 µm toward the free end" is a laser-frame artefact. Replace it with the
  lever-frame shift and the blind-fit prediction.
- R2 positions quoted "from the clamp" with the 6.1 µm offset are ~18 µm too large; use x − x₀(t).
- "The free end became less node-like / enhancement went stale" at stage 232 µm is mostly the laser
  drifting ~15 µm inward, not a change in the node.
- Positive message for the paper: a force curve at each stop already provides an *in-situ* position
  ruler (same idea as draft Sec. S1), so autonomous acquisition can correct laser drift as it goes.
"""),
("code", """
results.save("fig5_contact_state",
             dict(anchors={t: a.assign(t=a.t.astype(str)).to_dict("records") for t, a in an.items()},
                  nodes=nodes_tbl.assign(time=nodes_tbl.time.astype(str)).to_dict("records"),
                  fits={f"{k[0]}|{'nodes' if k[1] else 'blind'}": v for k, v in fits.items()},
                  node_prediction=cmp.to_dict("records")),
             table=cmp)
"""),
]
