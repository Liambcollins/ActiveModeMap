from nbbuild import SETUP

CELLS = [
("md", """
# SI-a · InvOLS drift and the absolute pm/V scale (Phase 2f)

Every checkpoint-based pm/V number in the SCM-PIT-B analysis converts raw lock-in volts with
**InvOLS(x) from the pre-flight walk**, measured once at the start of each campaign. The
R2 load-extension audit found that the calibration had aged ~10 % by the next morning and
put a blanket 10 % systematic on every absolute d33.

In fact each laser stop was followed by a force curve that re-measured InvOLS. Those values
are in the force-curve notes, so the calibration can be reconstructed at every stage and
position (`fmmpaper.calib`), and the reported numbers corrected with the InvOLS that applied
when and where the spectra were taken.

**Two traps handled by `calib.timeline`:**
- A force-curve file is time-stamped when the *next* curve starts. Its time therefore falls in
  the following position's interval (a one-position lag, shown below).
- Out-of-range retries (InvOLS above the pre-flight maximum) are rejected curves and are dropped.
"""),
("code", SETUP),
("code", """
import datetime as dt
from fmmpaper import calib
CAMPAIGNS = {"R1": ("DomainsB_SCMPIT", dt.datetime(2026, 9, 20, 11)),
             "R2": ("DomainsB_SCMPIT_R2", dt.datetime(2026, 9, 20, 22))}
tl = {tag: calib.timeline(c, d0) for tag, (c, d0) in CAMPAIGNS.items()}
naive = {tag: calib.timeline(c, d0, lag=0) for tag, (c, d0) in CAMPAIGNS.items()}
for tag, t in tl.items():
    print(tag, len(t), "force curves assigned to laser stops")
"""),
("md", "## Evidence for the one-position lag"),
("code", """
walks = ["10_bias_survey", "20_wideband_fast", "21_wideband_validation"]
fig, axs = fp.figure("double", aspect=0.3, ncols=2)
for ax, (lab, T) in zip(axs, (("as time-stamped (lag 0)", naive), ("lag-corrected (lag 1)", tl))):
    for tag, t in T.items():
        d = t[t.stage.isin(walks)]
        ax.plot(d.x_um - 6.1, d.ratio, "o", ms=3, color="0.55" if tag == "R1" else "k", label=tag)
    ax.set_ylim(0.3, 1.4); ax.axhline(1, color="0.8", lw=0.6)
    ax.set_title(lab); ax.set_xlabel("distance from clamp (µm)"); ax.set_ylabel("InvOLS / pre-flight InvOLS")
axs[0].legend()
fig.tight_layout(); fp.save(fig, "SIa_lag_evidence")
pd.DataFrame({tag: [naive[tag][naive[tag].stage.isin(walks)].ratio.std(),
                    tl[tag][tl[tag].stage.isin(walks)].ratio.std()] for tag in tl},
             index=["sd, lag 0", "sd, lag 1"]).round(3)
"""),
("md", "## The calibration timeline"),
("code", """
STAGE_C = {"bias_survey": "#1f77b4", "wideband_fast": "#ff7f0e", "wideband_validation": "#2ca02c",
           "dense_grid": "#d62728", "load_ladder": "#9467bd", "bias_vs_load": "#8c564b",
           "postcheck": "#e377c2", "quant_imaging": "#7f7f7f", "lowac_30mV": "#bcbd22",
           "ac_series": "#17becf", "closeout": "#000000"}
kind = lambda st: st.split("_", 1)[1]
fig, axs = fp.figure("double", aspect=0.32, ncols=2, sharey=True)
for ax, (tag, t) in zip(axs, tl.items()):
    for st, d in t.groupby("stage"):
        ax.plot(d.t, d.ratio, "o", ms=2, color=STAGE_C.get(kind(st), "0.5"), label=kind(st))
    ax.axhline(1, color="0.7", lw=0.6); ax.set_ylim(0.8, 1.6)
    ax.set_title(f"{tag}: InvOLS re-measured at every stop / pre-flight curve")
    ax.tick_params(axis="x", rotation=30)
axs[0].set_ylabel("ratio")
for ax in axs:
    ax.legend(fontsize=4.5, ncol=2, loc="upper left")
fig.tight_layout(); fp.save(fig, "SIa_invols_timeline")
factors = pd.concat({tag: calib.stage_factors(t) for tag, t in tl.items()})
factors.round(3)
"""),
("md", """
**Reading the timeline.**
- Within the first hour of each campaign the calibration has moved +2–8 %. This covers the
  bias survey (+2.3 % R2, +2.6 % R1), the live wideband capture and the validation.
- It keeps rising slowly: +6–9 % by the dense map and +10–21 % by the close-out.
- At a fixed position, repeated force curves agree to 0.6 % (AC series, 15–17 curves over
  35–40 min), so the scatter is not measurement noise.
- The ratio depends on position and on load. Along the load-ladder walks it rises from
  ~1.02 at the free end to 1.2–1.5 near the base, and it grows with load.
- Within each dense map (3.2–3.6 h, walked free end → base) the ratio climbs from ~1.0–1.04
  to 1.2–1.3. The 45-min bias-survey walk over the same positions is flat (0.99–1.03), so
  this climb is mainly drift in time, not a position effect. Dense-map spectra taken late,
  near the base, carry the largest factors. The static
  deflection shape itself changed, so a single scale factor cannot correct a whole stage.
  Per-position factors are used below.
"""),
("md", "## Correction 1: stage-1 d33 (bias survey), per position"),
("code", """
VAC = 1.0
corr = {}
for tag in ("R1", "R2"):
    s = F.load(f"scmpitB_{tag.lower()}_bias")
    _, _, ref = calib.preflight_curve(CAMPAIGNS[tag][0])
    f_meas = lambda x, _tl=tl[tag], _ref=ref: float(_ref(x) * calib.factor_at(_tl, "10_bias_survey", x)[0])
    t_old = domains.transfer_table(s, ref, VAC)
    t_new = domains.transfer_table(s, f_meas, VAC)
    corr[tag] = pd.DataFrame(dict(x_clamp_um=t_old.x_clamp_um, factor=t_new.invols / t_old.invols,
                                  d33_qs_preflight=t_old.d33_qs, d33_qs_measured=t_new.d33_qs))
    interior = t_old.x_um < t_old.x_um.max()
    print(f"{tag}: median d33_QS (interior) {np.median(t_old.d33_qs[interior]):.2f} -> "
          f"{np.median(t_new.d33_qs[interior]):.2f} pm/V; factor median "
          f"{np.median(corr[tag].factor):.3f}, range {corr[tag].factor.min():.3f}-{corr[tag].factor.max():.3f}")
pd.concat(corr).round(3)
"""),
("md", """
The blind-EB routes scale the same way: the per-position route by the per-position factor,
and the free-end route by the free-end factor. The E(x) ratios and the E ≠ Q argument are
unaffected, because they are ratios of two channels measured with the same InvOLS.
"""),
("md", "## Correction 2: load extension (R2 stage 41), d33 at position A"),
("code", """
s4 = F.load("scmpitB_r2_s4")
bvl = s4["bias_vs_load"]
fa = tl["R2"].query("stage == '41_bias_vs_load'").sort_values("t").ratio.values
loads = sorted(bvl, key=float)
rows = []
for L, f_ in zip(loads, fa):
    b = bvl[L]
    d_old = b["qs"]["P_abs"] * b["invols"] / 32 / VAC * 1e12
    rows.append(dict(load_nN=float(L), invols_used_um_per_V=b["invols"] * 1e6, factor_measured_at_A=f_,
                     d33_as_reported=d_old, d33_measured_invols=d_old * f_))
lx = pd.DataFrame(rows)
# stage-1 reference at A (148.8 um from clamp), interpolated between survey positions, corrected
c2 = corr["R2"]
d33_A_stage1 = float(np.interp(148.8, c2.x_clamp_um, c2.d33_qs_measured))
lx["vs_stage1_A_pct"] = 100 * (lx.d33_measured_invols / d33_A_stage1 - 1)
print(f"stage-1 d33_QS at A (measured InvOLS): {d33_A_stage1:.2f} pm/V")
lx.round(3)
"""),
("md", """
The audit's fix scaled the load-extension d33 by the *free-end* probe-check factor (1.10–1.17).
The factor measured *at position A* during that stage is 1.23–1.25, because the deflection
shape changed along the lever. With the correct factor, d33 at 300 and 500 nN sits about 4 %
above the stage-1 value at the same position, and 100 nN sits about 7 % below.
"""),
("md", "## Correction 3: comparing raw-volt quantities between R1 and R2 at position A"),
("code", """
absA = {}
for tag in ("R1", "R2"):
    _, _, ref = calib.preflight_curve(CAMPAIGNS[tag][0])
    f_ = calib.factor_at(tl[tag], "70_ac_series", [154.9])[0]
    absA[tag] = dict(preflight_invols_A=float(ref(154.9)), factor_ac_series=f_,
                     invols_A_during_ac=float(ref(154.9)) * f_)
absA = pd.DataFrame(absA).T
rA = absA.invols_A_during_ac["R1"] / absA.invols_A_during_ac["R2"]
print(f"InvOLS at A during the AC series, R1/R2 = {rA:.3f}: multiply an R1 raw-volt quantity by "
      f"{rA:.3f} (relative to R2) before comparing it with R2 in pm/V")
(absA * [1e6, 1, 1e6]).rename(columns={"preflight_invols_A": "preflight_invols_A_um_per_V",
                                       "invols_A_during_ac": "invols_A_during_ac_um_per_V"}).round(4)
"""),
("md", """
The R2 report compares AC-series slopes between campaigns *in raw volts* (7 % agreement).
The InvOLS at position A was not the same in the two series: R1's was about 7 % lower.
Any R1-vs-R2 comparison of raw-volt quantities should first be converted to pm/V with the
values above.
"""),
("md", """
## Most of the InvOLS "drift" is the laser moving along the lever
InvOLS(x) of a tip-loaded lever has the shape 1/[(x − x₀)²(3L − (x − x₀))] whatever the
contact stiffness. Fitting that shape to each walk separates the two things a changing InvOLS
can mean: a detector-gain change (overall scale) and a laser-position change (x₀). Notebook 05a
does this in full; the summary is below.
"""),
("code", """
L_UM = 226.5
anch = {tag: calib.frame_anchors(c, tl[tag], L_UM) for tag, (c, _) in CAMPAIGNS.items()}
pd.concat(anch).round(3)
"""),
("md", """
**Reading.**
- The gain stays within ±2–5 % of pre-flight all night. The clamp position x₀ in stage
  coordinates moves by ~1 µm/h, with jumps of 10–13 µm across the load ladder and the load
  extension.
- The per-stop InvOLS factors above remain the right numbers for pm/V: each force curve
  measures the sensitivity at the spot that is actually illuminated.
- What changes is **where** each spectrum was taken. Positions must be quoted as x − x₀(t),
  not x − 6.1 µm (see notebooks 05a and 06).
"""),
("md", """
## Uncertainty budget for absolute pm/V after correction

| term | size | source |
|---|---|---|
| force-curve repeatability at a fixed stop | 0.6 % (sd) | AC series, 15–17 curves per run |
| drift within a stage (between stops) | 1–3 % | spread of per-position factors in a walk |
| AmpInvOLS = InvOLS / 32 | confirmed (L. Collins, 2026-09-27) | the ratio is exactly 32.000 in all 606 notes, a software setting; checked independently for the Cypher QPDI |

The 10 % blanket systematic can be replaced by these per-stage, per-position factors. The ÷32
amplitude calibration has been confirmed independently, so no calibration item remains open.
"""),
("code", """
ratio32 = pd.concat([calib.fc_notes(c) for c, _ in CAMPAIGNS.values()])
r = ratio32.InvOLS / ratio32.AmpInvOLS
print(f"InvOLS/AmpInvOLS over {len(r)} notes: {r.mean():.4f} +- {r.std():.4f}")
results.save("si_invols_drift",
             dict(stage_factors={f"{a}|{b}": v for (a, b), v in factors["median"].items()},
                  d33_stage1_median_interior={t: float(np.median(corr[t].d33_qs_measured[:-1])) for t in corr},
                  d33_stage1_factor_median={t: float(np.median(corr[t].factor)) for t in corr},
                  load_extension=lx.to_dict("records"), invols_over_ampinvols=float(r.mean())),
             table=pd.concat(tl).reset_index(drop=True))
"""),
]
