from nbbuild import SETUP

CELLS = [
("md", """
# 04 · Live fast mode mapping on the instrument (paper Fig. 4)

Two probes, same protocol: an equispaced design measured in one pass, reconstructed
afterwards by a rank-r Chebyshev fit, and scored on a **separate capture** of positions
the design never visited (measured within the hour).

| probe | design | validation | reported |
|---|---|---|---|
| PPP-CONTAu (5 modes) | 12 pos, 13.8 min | 15 pos | 0.169 NRMSE, 98 % < 3 dB at rank 8 |
| SCM-PIT-B R2 (3 modes) | 8 pos, ~9 min | 7 midpoints | 0.19 NRMSE, 93 % < 3 dB at rank 5 |

The SCM-PIT-B numbers in the campaign reports came from a re-implementation (SVD + spline)
because scipy was missing on the instrument VM. Here both probes go through
`activemodemap.lowrank`, which is the Phase-2b re-score.
"""),
("code", SETUP),
("code", """
cases = {
    "PPP-CONTAu": ("ppp_wb_fast", "ppp_wb_valid", range(4, 12)),
    "SCM-PIT-B R2": ("scmpitB_r2_wb_fast", "scmpitB_r2_wb_valid", range(3, 8)),
}
if (F.config.data_root() / F.registry.DATASETS["scmpitB_r1_wb_fast"].path).exists():
    cases["SCM-PIT-B R1"] = ("scmpitB_r1_wb_fast", "scmpitB_r1_wb_valid", range(3, 8))

rows, keep = [], {}
for name, (kd, kv, ranks) in cases.items():
    d, v = F.load(kd), F.load(kv)
    assert np.allclose(d.freq_Hz, v.freq_Hz)
    for r in ranks:
        Zp, sd = recon.cross_capture(d.x_um, d.Z[0], v.x_um, arm="lowrank", rank=r)
        sc = recon.score(Zp, v.Z[0], d.freq_Hz, bands=d.probe.bands_Hz)
        rows.append(dict(probe=name, rank=r, N=len(d.x_um), **sc))
        keep[(name, r)] = (d, v, Zp)
live = pd.DataFrame(rows)
live.round(3)
"""),
("code", """
best = live.loc[live.groupby('probe').nrmse.idxmin()]
best[['probe', 'N', 'rank', 'nrmse', 'db_median', 'within_3dB', 'phase_mae_deg']].round(3)
"""),
("md", """
## R1 vs R2 on a matched span, in the lever frame (Phase 2b)

The R2 report partly attributed R2's better wideband score to a shorter span ("R2's pre-flight
stopped 20 µm earlier"). In stage coordinates R2 starts at 72 µm, against 52 µm for R1. But the
clamp was at x₀ = 5.95 µm during R1's wideband stage and 23.95 µm during R2's (notebook 05a).
In the lever frame both runs start at 46–48 µm, and it is R2 that stops 18 µm short of the tip
(208 vs 226 µm).

The fair test keeps all 8 design positions and scores only validation positions inside the shared
lever span. Dropping design points instead would confound span with N. Validation positions use
x₀ at the validation stage, which is 2.75 µm (R1) and 1.1 µm (R2) later than the design stage.
"""),
("code", """
import datetime as dt
from fmmpaper import calib
CAMP = {"R1": ("DomainsB_SCMPIT", dt.datetime(2026, 9, 20, 11)),
        "R2": ("DomainsB_SCMPIT_R2", dt.datetime(2026, 9, 20, 22))}
X0 = {}
for t, (c, d0) in CAMP.items():
    a = calib.frame_anchors(c, calib.timeline(c, d0), 226.5).set_index("stage")
    X0[t] = (float(a.loc["20_wideband_fast", "x0"]), float(a.loc["21_wideband_validation", "x0"]))
runs = {t: (F.load(f"scmpitB_{t.lower()}_wb_fast"), F.load(f"scmpitB_{t.lower()}_wb_valid"))
        for t in CAMP if (F.config.data_root() / F.registry.DATASETS[f"scmpitB_{t.lower()}_wb_fast"].path).exists()}
lo = max(d.x_um.min() - X0[t][0] for t, (d, v) in runs.items())
hi = min(d.x_um.max() - X0[t][0] for t, (d, v) in runs.items())
print({t: tuple(round(x, 2) for x in v) for t, v in X0.items()}, f"shared lever span {lo:.1f}-{hi:.1f} um")
rows = []
for t, (d, v) in runs.items():
    xd, xv = d.x_um - X0[t][0], v.x_um - X0[t][1]
    for frame, xdd, xvv in (("stage", d.x_um, v.x_um), ("lever", xd, xv)):
        for tag, mv in (("all validation", np.ones(len(xv), bool)),
                        ("shared span only", (xv >= lo) & (xv <= hi))):
            for r in (4, 5, 6):
                Zp, _ = recon.cross_capture(xdd, d.Z[0], xvv[mv], arm="lowrank", rank=r)
                sc = recon.score(Zp, v.Z[0][mv], d.freq_Hz, bands=d.probe.bands_Hz)
                rows.append(dict(run=t, frame=frame, scored=tag, n_valid=int(mv.sum()), rank=r,
                                 **{k: sc[k] for k in ("nrmse", "within_3dB", "phase_mae_deg",
                                                       "nrmse_CR1", "nrmse_CR2", "nrmse_CR3")}))
matched = pd.DataFrame(rows)
matched.pivot_table(index=["frame", "scored", "rank"], columns="run",
                    values=["nrmse", "nrmse_CR3"]).round(1)
"""),
("md", """
**Reading.**
- Correcting for the clamp moving between the design and validation captures changes the
  error by less than 2 points. At 23–26 µm spacing this correction is negligible.
- On the shared span R1 is still worse: 26.8 % vs 18.9 % at rank 5. R1's two tip-end validation
  positions are its best (21 %), not its worst.
- **The span confound is withdrawn.** R2 reconstructs better because of the contact state, not
  the span. The gap is almost all CR3 (43 % vs 23 %), consistent with R2's higher CR3 Q
  (271 vs 138, notebook 03a) after the R1 load ladder changed the tip (notebook 05a).
- The package-backed numbers agree with the report's re-implementation to within 1–2 points
  (R1 26.5 %, R2 19.3 % at rank 5).
"""),
("code", """
fig, axs = plt.subplots(len(cases), 3, figsize=(fp.WIDTH_IN['double'], 2.2 * len(cases)))
axs = np.atleast_2d(axs)
for row, name in enumerate(cases):
    b = best.query("probe == @name").iloc[0]
    d, v, Zp = keep[(name, int(b['rank']))]
    ref = np.abs(v.Z[0]).max()
    fp.map_db(axs[row, 0], v.x_um, v.freq_Hz, v.Z[0], ref=ref, colorbar=False)
    axs[row, 0].set_title(f"{name}: measured at {len(v.x_um)} held-out positions", fontsize=7)
    fp.map_db(axs[row, 1], v.x_um, v.freq_Hz, Zp, ref=ref, colorbar=False)
    fp.mark_positions(axs[row, 1], d.x_um)
    axs[row, 1].set_title(f"predicted from {len(d.x_um)} (rank {int(b['rank'])})", fontsize=7)
    j = len(v.x_um) // 2
    fp.spectrum(axs[row, 2], v.freq_Hz, v.Z[0, j], label='measured', db=True, lw=0.8)
    fp.spectrum(axs[row, 2], v.freq_Hz, Zp[j], label='predicted', color=fp.C['lowrank'], db=True, lw=0.8, ls='--')
    axs[row, 2].set_title(f"never-visited x = {v.x_um[j]:.1f} µm", fontsize=7)
    axs[row, 2].set_ylabel("|Z| (dB V)"); axs[row, 2].legend(fontsize=6)
fig.tight_layout(); fp.save(fig, "04_live_fmm")
"""),
("code", """
fig, ax = fp.figure("single", aspect=0.7)
for name in cases:
    d = live.query("probe == @name")
    ax.plot(d['rank'], d.nrmse / 100, 'o-', ms=3, label=name)
ax.set_xlabel("rank"); ax.set_ylabel("held-out complex NRMSE"); ax.legend(fontsize=6)
fp.save(fig, "04_error_vs_rank")
results.save("fig4_live_fmm", {r.probe: dict(rank=int(r['rank']), nrmse=r.nrmse, within_3dB=r.within_3dB,
                                              phase_mae_deg=r.phase_mae_deg) for _, r in best.iterrows()},
             table=live)
"""),
]
