from nbbuild import SETUP

CELLS = [
("md", """
# 00 · Data overview

What is on disk, in which convention, and what each dataset looks like.
Run this first on a new machine: the `on_disk` column tells you whether the
data root is set correctly.

**Conventions (enforced by `fmmpaper.io`)**
- `Z` is complex, raw lock-in volts for checkpoints (× InvOLS / 32 → metres).
- Phase is in the model convention (falls through resonance). Pre-2026-09-17 files are Igor-raw and are conjugated on load.
- Positions are stage µm; distance from clamp = `x_um − probe.clamp_offset_um`.
"""),
("code", SETUP),
("md", "## Registry: probes and datasets"),
("code", """
pd.DataFrame([dict(key=p.key, name=p.name, k_N_per_m=p.k_N_per_m, L_um=p.L_um,
                   clamp_offset_um=p.clamp_offset_um,
                   bands_kHz={k: tuple(round(v/1e3) for v in b) for k, b in p.bands_Hz.items()})
              for p in F.registry.PROBES.values()])
"""),
("code", """
avail = F.available()
avail
"""),
("md", "## Shape, span and conditions of every series on disk"),
("code", """
rows = []
for key in avail.query("on_disk and kind in ['series', 'dense']").key:
    try:
        s = F.load(key)
    except Exception as e:
        rows.append(dict(key=key, error=str(e))); continue
    if isinstance(s, dict):              # folder datasets (load ladder, AC series)
        rows.append(dict(key=key, shape=f"{len(s)} checkpoints", conditions=list(s)[:6]))
        continue
    rows.append(dict(key=key, shape=s.Z.shape, x_span=f"{s.x_um.min():.1f}-{s.x_um.max():.1f}",
                     f_kHz=f"{s.freq_Hz.min()/1e3:.1f}-{s.freq_Hz.max()/1e3:.0f}",
                     n_freq=s.freq_Hz.size,
                     phase_file=s.meta['phase_convention_file'],
                     conjugated=s.meta['phase_converted'],
                     conditions={c: sorted(s.conditions[c].dropna().unique().tolist())
                                 for c in s.conditions.columns}))
pd.DataFrame(rows)
"""),
("md", """
## Phase-convention check
After loading, the phase through the CR1 peak should *fall* (negative slope) for every dataset.
A positive number means a file was mislabelled and phase-sensitive results from it are wrong.
"""),
("code", """
chk = []
for key in avail.query("on_disk and kind in ['series', 'dense']").key:
    s = F.load(key)
    if isinstance(s, dict):
        continue
    band = s.probe.bands_Hz['CR1']
    j = s.Z.shape[1] // 2
    chk.append(dict(key=key, x_um=s.x_um[j],
                    dphase_deg=io.phase_slope_through_peak(s.freq_Hz, s.Z[0, j], band)))
pd.DataFrame(chk)
"""),
("md", "## Quick-look maps: one per probe"),
("code", """
fig, axs = fp.figure("double", aspect=0.36, ncols=3)
g = F.load("scmpitA_gridB")
fp.map_db(axs[0], g.meta['x_true_um'], g.freq_Hz, g.Z[0], fmax=1200)
axs[0].set_title("SCM-PIT-A · Dense_Grid_B (241 pos, 4.7 h)")
axs[0].set_xlabel("position, InvOLS-ruler axis (µm)")

s = F.load("scmpitB_r2_bias"); i0 = s.where(bias_V=0, spot=1)[0]
fp.map_db(axs[1], s.x_clamp_um, s.freq_Hz, s.Z[i0], fmax=2000)
fp.mark_positions(axs[1], s.x_clamp_um)
axs[1].set_title("SCM-PIT-B · R2 bias survey, 0 V, spot 1")
axs[1].set_xlabel("distance from clamp (µm)")

p = F.load("ppp_wb_fast")
fp.map_db(axs[2], p.x_um, p.freq_Hz, p.Z[0], fmax=1100)
fp.mark_positions(axs[2], p.x_um)
axs[2].set_title("PPP-CONTAu · live 12-position capture")
fig.tight_layout()
fp.save(fig, "00_quicklook_three_probes")
"""),
("md", """
## InvOLS position ruler (SCM-PIT-A)
Draft Sec. S1: x_true ≈ 0.843 x_cmd + 37.5 µm with ~0.5 µm rms. Reproduced here from the log.
"""),
("code", """
print("axis fit (slope, offset):", np.round(g.meta['axis_fit'], 4), " rms", round(g.meta['axis_rms_um'], 3), "um")
fig, ax = fp.figure("single", aspect=0.7)
ax.plot(g.x_um, g.meta['x_true_um'], 'k-', label='dense sweep (InvOLS ruler)')
ax.plot(g.meta['anchors_x_um'], g.meta['anchors_x_um'], 'o', color=fp.C['ebgp'], ms=3, label='large-move anchors')
ax.plot(g.x_um, g.x_um, ':', color='0.6', label='commanded = true')
ax.set_xlabel("commanded position (µm)"); ax.set_ylabel("true position (µm)"); ax.legend()
fp.save(fig, "00_gridB_axis")
"""),
]
