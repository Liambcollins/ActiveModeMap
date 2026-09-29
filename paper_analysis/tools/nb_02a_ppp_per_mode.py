from nbbuild import SETUP

CELLS = [
("md", """
# 02a · Per-mode recovery on the soft probe: EB, EB+GP, GP and low-rank (Phase 2a, feeds Fig. 3)

**Question.** Does physics win at small N on every mode, and does the N window where it wins
widen with the number of nodes in the span?

**Data.** PPP-CONTAu 1 µm dense map: 346 positions, 100–445 µm, 100 nN, 0 V, PFM drive,
zero NaN (`ppp_dense_1um`). This replaces the 5 µm, 75-position sweep used on 2026-09-19
(`softprobe-recovery-vs-N-status`). That sweep resolved CR5's nodes (90 µm apart) only
marginally.

**Arms.** Every arm sees only the selected positions and is scored on the held-out ones.
- **EB**: `activemodemap.inference.PhysicsPosterior` with ζ fitted, a complex gain solved
  per evaluation, the tip setback fitted and f0 fixed at the measured 13.649 kHz, via
  `fmmpaper.physrec`. This is the 2026-09-17 package (branch
  `feat/position-scale-and-reanalysis`).
- **EB+GP**: the EB fit plus the package's discrepancy GP (`HybridSurrogate`).
- **GP**: model-free, PRESS length scale (`recon.rec_gp`, the draft's Grid B arm).
- **Low-rank**: Chebyshev basis, rank ≤ 6 (`recon.rec_lowrank`). The live protocol uses the
  same rank.

Designs are equispaced over the span, including both ends (the Grid B protocol).

**Runtime.** About 70–80 s per EB fit, so ~1 h for 5 modes × 9 budgets. Results are
cached per (mode, N) in `results/_cache_02a.json`, and a rerun resumes where it stopped.
"""),
("code", SETUP),
("code", """
import json, time
from fmmpaper import physrec
s = F.load("ppp_dense_1um")
x = s.x_um - s.probe.clamp_offset_um        # stage frame; no static-shape anchor for this probe yet
BANDS = ["CR1", "CR2", "CR3", "CR4", "CR5"]
NS = [3, 4, 5, 6, 8, 10, 14, 20, 30]
GEOM = physrec.GEOM["pppcontau"]
band_data = {b: physrec.band_slice(s.freq_Hz, s.Z[0], s.probe.bands_Hz[b], n_max=250) for b in BANDS}
print(s)
print({b: (z.shape, round(f[0] / 1e3, 1), round(f[-1] / 1e3, 1)) for b, (z, f) in band_data.items()})
"""),
("md", "## Ground truth: resonance, Q and nodes of each mode on the dense map"),
("code", """
def nodes_of(Zb, fb):
    pk = spectra.peaks_along_x(fb, Zb, (fb[0], fb[-1]))
    return spectra.nodes_from_profile(x, pk["amp"]), pk

truth = {}
rows = []
for b, (zb, fb) in band_data.items():
    nd, pk = nodes_of(zb, fb)
    # score only nodes >= 5 um inside the span: the node finder cannot see a minimum at the edge
    truth[b] = [n for n in nd if x.min() + 5 <= n <= x.max() - 5]
    rows.append(dict(mode=b, f_kHz=np.median(pk["f_Hz"]) / 1e3, Q=np.nanmedian(pk["Q"]),
                     n_nodes=len(nd), nodes_um=np.round(nd, 1).tolist(), scored=np.round(truth[b], 1).tolist()))
pd.DataFrame(rows).round(2)
"""),
("code", """
fig, axs = fp.figure("double", aspect=0.5, ncols=5, sharey=False)
for ax, (b, (zb, fb)) in zip(axs, band_data.items()):
    fp.map_db(ax, x, fb, zb)
    for n in truth[b]:
        ax.axvline(n, color="w", lw=0.5, ls=":")
    ax.set_title(b); ax.set_xlabel("x (µm)")
axs[0].set_ylabel("frequency (kHz)")
fig.tight_layout(); fp.save(fig, "02a_ppp_truth_bands")
"""),
("md", "## Learning curves (cached; about 1 h on first run)"),
("code", """
CACHE = F.config.RESULTS_DIR / "_cache_02a.json"
cache = json.loads(CACHE.read_text()) if CACHE.exists() else {}

def held(sel):
    return recon.held_out(len(x), sel)

def err(M, zb, h):
    return float(np.sqrt(np.mean(np.abs(M[h] - zb[h]) ** 2)) / np.sqrt(np.mean(np.abs(zb[h]) ** 2)))

def node_err(M, fb, true_nodes):
    nd, _ = nodes_of(M, fb)
    if not true_nodes:
        return np.nan, len(nd)
    if not nd:
        return np.inf, 0
    return float(np.mean([min(abs(t - m) for m in nd) for t in true_nodes])), len(nd)

for b, (zb, fb) in band_data.items():
    for n in NS:
        key = f"{b}|{n}"
        if key in cache:
            continue
        sel = recon.select_equispaced(x, n); h = held(sel)
        t0 = time.time()
        o = physrec.rec_eb(x, sel, zb[sel], fb, GEOM)
        maps = {"EB": o["Zeb"], "EB+GP": o["Zrec"],
                "GP": recon.rec_gp(x, sel, zb[sel])["Zrec"],
                "low-rank": recon.rec_lowrank(x, sel, zb[sel], rank=min(n, 6))["Zrec"]}
        rec = {"theta": o["theta"], "sec": time.time() - t0, "sel": [int(i) for i in sel]}
        for arm, M in maps.items():
            ne, nn = node_err(M, fb, truth[b])
            rec[arm] = dict(nrmse=err(M, zb, h), node_err_um=ne, n_nodes=nn)
        cache[key] = rec
        CACHE.write_text(json.dumps(cache))
        print(f"{key:7s} {rec['sec']:5.0f}s  " + "  ".join(f"{a} {rec[a]['nrmse']:.3f}" for a in maps))

ARMS = ["EB", "EB+GP", "GP", "low-rank"]
lc = pd.DataFrame([dict(mode=k.split("|")[0], N=int(k.split("|")[1]), arm=a, **v[a])
                   for k, v in cache.items() for a in ARMS])
lc.pivot_table(index=["mode", "N"], columns="arm", values="nrmse").round(4)
"""),
("code", """
COL = {"EB": fp.C["eb"] if "eb" in fp.C else "0.4", "EB+GP": fp.C["ebgp"], "GP": fp.C["gp"],
       "low-rank": fp.C["lowrank"]}
fig, axs = fp.figure("double", aspect=0.55, ncols=5, nrows=2, sharex=True)
for j, b in enumerate(BANDS):
    d = lc[lc["mode"] == b]
    for a in ARMS:
        da = d[d.arm == a].sort_values("N")
        axs[0, j].plot(da.N, da.nrmse, "o-", ms=2.5, color=COL[a], label=a)
        ne = np.maximum(da.node_err_um.replace(np.inf, np.nan), 0.5)   # 0 -> below the 1 um grid
        axs[1, j].plot(da.N, ne, "o-", ms=2.5, color=COL[a])
    axs[0, j].set_yscale("log"); axs[0, j].set_title(f"{b} ({len(truth[b])} scored nodes)")
    if np.isfinite(d.node_err_um.replace(np.inf, np.nan)).any() and (d.node_err_um > 0).any():
        axs[1, j].set_yscale("log")
    else:
        axs[1, j].text(0.5, 0.5, "no interior node", transform=axs[1, j].transAxes, ha="center", fontsize=6)
    fp.n_axis(axs[1, j])
    axs[1, j].set_xlabel("measured positions N")
axs[0, 0].set_ylabel("held-out complex NRMSE"); axs[1, 0].set_ylabel("mean node error (µm; 0.5 = on grid)")
axs[0, 0].legend(fontsize=5.5)
fig.tight_layout(); fp.save(fig, "02a_ppp_per_mode_learning_curves")
"""),
("md", "## Crossover: smallest N at which a data-driven arm matches the physics arm"),
("code", """
def crossover(d, phys="EB+GP", data_arm="low-rank"):
    # smallest N from which the data-driven arm stays at or below the physics arm for every larger N;
    # robust to budgets where every arm fails (both ~1, order is noise)
    p = d[d.arm == phys].set_index("N").nrmse
    q = d[d.arm == data_arm].set_index("N").nrmse
    ns = [n for n in NS if n in p.index]
    for i, n in enumerate(ns):
        if all(q[m] <= p[m] for m in ns[i:]):
            return n
    return f"never (to N={ns[-1]})"

xo = pd.DataFrame([dict(mode=b, n_nodes=len(truth[b]),
                        lowrank_vs_EBGP=crossover(lc[lc["mode"] == b]),
                        GP_vs_EBGP=crossover(lc[lc["mode"] == b], data_arm="GP"),
                        lowrank_vs_EB=crossover(lc[lc["mode"] == b], phys="EB"))
                   for b in BANDS])
xo
"""),
("code", """
th = pd.DataFrame([dict(mode=k.split("|")[0], N=int(k.split("|")[1]), **{p: v["theta"].get(p) for p in
                   ("k_ratio", "zeta", "tip_setback_um", "red_chi2")}) for k, v in cache.items()])
th.groupby("mode")[["tip_setback_um", "zeta", "k_ratio"]].agg(["median", "min", "max"]).round(4)
"""),
("md", """
**Reading guide.** Fill in the text once the numbers are in.
- 2026-09-19 result on the 5 µm sweep: CR1 crossover at N = 4 (physics leads again from N = 14);
  CR2 no crossover up to N = 20.
- Check whether the fitted setback and ζ agree across modes. One lever has one setback: a
  mode-dependent setback means the model is absorbing a missing mechanism.
- Node errors at CR4 and CR5 are the part the 1 µm grid makes possible.
"""),
("code", """
results.save("fig3_ppp_per_mode", dict(truth_nodes=truth, crossover=xo.to_dict("records"),
             geom=GEOM, NS=NS), table=lc)
"""),
]
