"""Bring notebooks/07_pfm_image_two_domain_bias.ipynb in line with the 2026-09-18 plan:
sparse 8-position survey + free-end cluster instead of an AL/dense map, interleaved
bias order, tune-window + phase-convention + higher-mode checks after the first tune,
a first-position go/no-go cell, and per-mode / null-vs-bias analysis."""
import json, sys, copy
p = sys.argv[1]
nb = json.load(open(p, encoding='utf-8'))
cells = nb['cells']


def src(i): return ''.join(cells[i]['source'])
def setsrc(i, s): cells[i]['source'] = s.splitlines(keepends=True)
def code(s): return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': s.splitlines(keepends=True)}
def md(s): return {'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(keepends=True)}
def find(startswith):
    for i, c in enumerate(cells):
        if src(i).lstrip().startswith(startswith): return i
    raise KeyError(startswith)


# ---- header cell: state the design -----------------------------------------
i = find('# 07 · PFM image')
s = src(i)
if '2026-09-18 design' not in s:
    s += '''
---
**2026-09-18 design (supersedes the AL loop below in §7).** Not a dense map vs bias. Two fixed
designs, run in this order:

1. **Sparse survey** — 8 equispaced laser positions over 75–445 µm, both domains × 9 biases
   (18 tunes/position, ~5 min). Gives the electrostatic/piezo ratio per mode, V_cpd, and the
   wideband reconstruction test (low-rank rank ≈ N/2 recovers 5–995 kHz from ≥ 8 positions).
2. **Free-end cluster** — 420–445 µm at 2 µm pitch, one domain × 5 biases. The CR1 null
   (12.6 µm from the tip at 0 V) is predicted to move with V_dc if the electrostatic drive
   dominates CR1; CR3's should not.

Between them a **go/no-go after position 1**: CR1 should swing hard with bias and CR3–CR5
barely; CR3–CR5 should flip ~180° between domains and CR1 much less. If CR3 behaves like CR1
the mixture picture is wrong — stop and look before spending the remaining hours.
Keep the full 5–995 kHz × 32000-point tune throughout; the higher modes are the piezo-clean channel.
'''
    setsrc(i, s)

# ---- parameters ------------------------------------------------------------
i = find('file_loc')
s = src(i)
s = s.replace("BIAS_V  = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]",
              "BIAS_V  = [-4.0, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 4.0]\n"
              "BIAS_ORDER = 'interleaved'            # 0, +4, -4, +3, -3, ... : drift becomes scatter, not slope")
s = s.replace("CHECKPOINT = os.path.join(file_loc, 'domains_bias_checkpoint.npz')",
'''CHECKPOINT = os.path.join(file_loc, 'domains_bias_checkpoint.npz')

# --- the 2026-09-18 design: sparse survey + free-end cluster --------------------
SPARSE_X_UM    = np.round(np.linspace(X_LO_UM, PROBE_L_UM, 8), 1)   # 75 ... 445, 8 positions
CLUSTER_X_UM   = np.arange(420.0, PROBE_L_UM + 1e-6, 2.0)            # CR1 null region, 2 um pitch
CLUSTER_BIAS_V = [-4.0, -2.0, 0.0, 2.0, 4.0]
CLUSTER_SPOTS  = [SPOT_UP]                                            # add SPOT_DOWN if time allows (+25 min)
CLUSTER_CHECKPOINT = os.path.join(file_loc, 'cluster_bias_checkpoint.npz')

# --- what the tune must look like (checked after the first tune) -----------------
TUNE_MIN_SPAN_HZ = 900e3      # 5-995 kHz window
TUNE_MIN_POINTS  = 30000      # 32000-point tune, ~10.5 s
MODES_KHZ = [63.47, 200.12, 408.02, 667.34, 916.14]   # CR1-CR5 from the dense sweep
MODE_HALF_WIN_HZ = 4e3''')
s = s.replace('''n_cond = len(BIAS_V) * len(LOAD_NN) * 2
print(f'{n_cond} conditions per laser position ({len(BIAS_V)} biases x 2 domains)')
print(f'~{n_cond * 17 / 60:.1f} min of tuning per position + ~1 min setup;')
print(f'{MIN_POSITIONS}-{MAX_POSITIONS} positions -> {n_cond*17/60*MIN_POSITIONS/60:.1f}'
      f'-{n_cond*17/60*MAX_POSITIONS/60:.1f} h')''',
'''T_TUNE_S = 10.5 + 2.0        # 32000-pt tune + bias settle
n_cond = len(BIAS_V) * len(LOAD_NN) * 2
t_sparse  = len(SPARSE_X_UM) * (n_cond * T_TUNE_S + 60 + 2 * 30) / 60       # + setup + 2 spot hops
t_cluster = len(CLUSTER_X_UM) * (len(CLUSTER_BIAS_V) * len(CLUSTER_SPOTS) * T_TUNE_S + 60) / 60
print(f'sparse survey : {len(SPARSE_X_UM)} positions x {n_cond} conditions  ~{t_sparse:.0f} min')
print(f'free-end cluster: {len(CLUSTER_X_UM)} positions x {len(CLUSTER_BIAS_V) * len(CLUSTER_SPOTS)} conditions  ~{t_cluster:.0f} min')
print(f'total ~{(t_sparse + t_cluster) / 60:.1f} h (plus image + domain finding ~30 min)')''')
setsrc(i, s)

# ---- after the first tune: window, convention, higher modes ----------------------
i = find('inst.a.do_autowedge()')
check = '''# --- checks on the first tune: window, phase convention, higher modes -----------
from activemodemap.asylum import check_phase_convention, tune_to_complex
td = tune['tune_data']
f_td = np.asarray(td['frequency'], float); good = np.isfinite(f_td) & np.isfinite(td['amplitude'])
span = f_td[good].max() - f_td[good].min(); npts = int(good.sum())
print(f'tune window {f_td[good].min()/1e3:.1f}-{f_td[good].max()/1e3:.1f} kHz, {npts} finite points')
assert span >= TUNE_MIN_SPAN_HZ and npts >= TUNE_MIN_POINTS, \\
    'set the Tune panel to 5-995 kHz x 32000 points (it was yesterday); the higher modes are needed'
assert abs(f_cr - CR_GUESS_HZ) < 500, f'CR1 at {f_cr/1e3:.2f} kHz, expected ~63.5 - contact or probe changed?'

# the Cypher phase must RISE through the peak (+158 deg yesterday); PHASE_SIGN=-1 conjugates it.
dphi = check_phase_convention(td, expect_sign=+1.0)
assert np.isfinite(dphi) and dphi > 45, f'raw phase slope {dphi:+.0f} deg - convention differs from yesterday, stop'
f_z, Z1 = tune_to_complex(td, check_convention=False)
A1 = np.abs(Z1); floor = np.median(A1[f_z > 950e3])
print('mode   f_peak   peak/floor (dB)')
for fk in MODES_KHZ:
    w = np.abs(f_z - fk * 1e3) <= MODE_HALF_WIN_HZ
    j = np.argmax(A1[w]); print(f'{fk:7.1f}  {f_z[w][j]/1e3:7.2f}  {20*np.log10(A1[w][j]/floor):5.1f}')
# CR1 sits at the tip end here; CR3+ should still be >= 15 dB above the floor at x = 445
'''
cells.insert(i + 1, code(check))

# ---- §7: sparse survey, first position only ------------------------------------
i = find('conditions = make_conditions(BIAS_V')
setsrc(i, '''conditions = make_conditions(BIAS_V, LOAD_NN, spots=[SPOT_UP, SPOT_DOWN], bias_order=BIAS_ORDER)
print(f'{len(conditions)} conditions/position:', [str(c) for c in conditions])

# Fixed design, free end first (the laser is parked there after imaging). Position 1 only:
# the next cell is the go/no-go before the remaining seven.
SPARSE_ORDER = SPARSE_X_UM[::-1]
series = run_series(inst, SPARSE_X_UM, conditions, ref_index=REF_INDEX, rank=RANK,
                    dns_band_Hz=DNS_BAND_HZ, checkpoint_path=CHECKPOINT, resume=True,
                    positions_um=SPARSE_ORDER[:1], verbose=True)
''')

gonogo_md = md('''### Go / no-go after position 1

Prediction from the wideband ladder (`pppcontau-electrostatic-vs-contact-drive-2026-09-18`):
electrostatic/piezo ≈ 4 at CR1, ≈ 0.6 at CR2, ≈ 0.06 at CR5. So **CR1 swings hard with bias and
CR3–CR5 barely**, and at 0 V **CR3–CR5 flip ~180° between domains while CR1 flips much less**.
If CR3 swings like CR1, or all modes flip identically, the mixture picture is wrong — stop here.''')
gonogo = code('''def mode_peak(Zrow, f, fk_khz, half=MODE_HALF_WIN_HZ):
    w = np.abs(f - fk_khz * 1e3) <= half
    j = int(np.argmax(np.abs(Zrow[w]))); return Zrow[w][j], f[w][j]

def bias_table(series, conditions, x_index=-1):
    """Per mode and spot: |Z| at the mode peak vs bias, its swing, and the 0 V domain flip."""
    f = series['freq_Hz']; out = {}
    for fk in MODES_KHZ:
        row = {}
        for spot in (SPOT_UP, SPOT_DOWN):
            V, A, Z0 = [], [], None
            for i, c in enumerate(conditions):
                if c.spot != spot: continue
                z, _ = mode_peak(series['Z'][i, x_index], f, fk)
                V.append(c.bias_V); A.append(abs(z))
                if abs(c.bias_V) < 1e-9: Z0 = z
            V, A = np.array(V), np.array(A); o = np.argsort(V)
            row[spot] = dict(V=V[o], A=A[o], swing=(A.max() - A.min()) / A[np.argmin(np.abs(V))], Z0=Z0)
        flip = np.degrees(np.angle(row[SPOT_UP]['Z0'] * np.conj(row[SPOT_DOWN]['Z0'])))
        out[fk] = dict(row, flip_deg=flip)
    return out

tab = bias_table(series, conditions)
print(f"x = {series['x_um'][-1]:.0f} um")
print('mode      swing(up)  swing(down)   0 V domain flip')
for fk, r in tab.items():
    print(f'{fk:7.1f}  {r[SPOT_UP]["swing"]:8.2f}  {r[SPOT_DOWN]["swing"]:9.2f}     {abs(r["flip_deg"]):6.0f} deg')
fig, axs = plt.subplots(1, len(MODES_KHZ), figsize=(3.2 * len(MODES_KHZ), 3), sharey=False)
for ax, (fk, r) in zip(axs, tab.items()):
    for spot, st in ((SPOT_UP, 'o-'), (SPOT_DOWN, 's--')):
        ax.plot(r[spot]['V'], r[spot]['A'], st, label=f'spot {spot}')
    ax.set_title(f'{fk:.0f} kHz  flip {abs(r["flip_deg"]):.0f} deg', fontsize=9); ax.set_xlabel('V_dc (V)')
axs[0].set_ylabel('|Z| at mode peak'); axs[0].legend(fontsize=7, frameon=False)
plt.tight_layout(); plt.savefig(os.path.join(file_loc, 'position1_go_nogo.png'), dpi=160); plt.show()
print('GO if: CR1 swing >> CR3-CR5 swing, and CR3-CR5 flip ~180 deg while CR1 flips much less.')
''')
cont_md = md('''### Remaining seven sparse positions (resumes from the checkpoint; position 1 is not repeated)''')
cont = code('''series = run_series(inst, SPARSE_X_UM, conditions, ref_index=REF_INDEX, rank=RANK,
                    dns_band_Hz=DNS_BAND_HZ, checkpoint_path=CHECKPOINT, resume=True,
                    positions_um=SPARSE_ORDER, verbose=True)
print(f"\\n{series['x_um'].size} positions x {len(conditions)} conditions x {series['freq_Hz'].size} frequencies")
''')
cluster_md = md('''## 7b · Free-end cluster — does the CR1 null move with bias?

420–445 µm at 2 µm pitch, one domain, 5 biases, interleaved. Its own checkpoint. Add
`SPOT_DOWN` to `CLUSTER_SPOTS` for the two-domain version if time allows.''')
cluster = code('''conditions_c = make_conditions(CLUSTER_BIAS_V, LOAD_NN, spots=CLUSTER_SPOTS, bias_order=BIAS_ORDER)
print(f'{len(conditions_c)} conditions/position:', [str(c) for c in conditions_c])
cluster = run_series(inst, CLUSTER_X_UM, conditions_c, ref_index=0, rank=2,
                     dns_band_Hz=DNS_BAND_HZ, checkpoint_path=CLUSTER_CHECKPOINT, resume=True,
                     positions_um=CLUSTER_X_UM[::-1], verbose=True)
inst.close()
print(f"\\ncluster: {cluster['x_um'].size} positions x {len(conditions_c)} conditions")
''')
cells[i + 1:i + 1] = [gonogo_md, gonogo, cont_md, cont, cluster_md, cluster]

# ---- §8 route 2: use a fitted V_cpd, not 0 ------------------------------------------
i = find('# route 1 -- domain difference at 0 V')
s = src(i)
s = s.replace('''ch_fit = separate_channels(series_up, load_nN=LOAD_NN[0], v_cpd=0.0)
sp_fit = channel_spots(ch_fit, x_grid, rank=RANK, band_Hz=DNS_BAND_HZ)
print('\\nV_cpd proxy from the fit: %.3f V' % estimate_v_cpd(ch_fit, band_Hz=DNS_BAND_HZ))''',
'''# v_cpd = 0 leaves a -V_cpd*b electrostatic term in the "piezo" channel (SCM-PIT lesson:
# with V_cpd = -1.9 V that term was 97-108 % of the channel). Estimate, then re-separate.
ch_0 = separate_channels(series_up, load_nN=LOAD_NN[0], v_cpd=0.0)
V_CPD = float(estimate_v_cpd(ch_0, band_Hz=DNS_BAND_HZ))
print(f'\\nV_cpd estimate (CR1 band): {V_CPD:+.3f} V   (SCM-PIT/PPLN gave -1.9 V at resonance)')
ch_fit = separate_channels(series_up, load_nN=LOAD_NN[0], v_cpd=V_CPD)
sp_fit = channel_spots(ch_fit, x_grid, rank=RANK, band_Hz=DNS_BAND_HZ)''')
s = s.replace("sp_dom = channel_spots(ch_dom, x_grid, rank=RANK, band_Hz=DNS_BAND_HZ)",
              "sp_dom = channel_spots(ch_dom, SPARSE_X_UM, rank=min(RANK, len(series['x_um']) - 1), band_Hz=DNS_BAND_HZ)")
s = s.replace("sp_fit = channel_spots(ch_fit, x_grid, rank=RANK, band_Hz=DNS_BAND_HZ)",
              "sp_fit = channel_spots(ch_fit, SPARSE_X_UM, rank=min(RANK, len(series['x_um']) - 1), band_Hz=DNS_BAND_HZ)")
setsrc(i, s)

# ---- §8: per-mode electrostatic / piezo ratio and null vs bias --------------------
i = find('# Bias dependence per domain')
setsrc(i, '''# Bias dependence per domain at each mode's OWN peak (not a fixed bin: the peak moves a
# few hundred Hz with bias and a fixed bin turns that into fake non-linearity).
fig, axs = plt.subplots(1, len(MODES_KHZ), figsize=(3.2 * len(MODES_KHZ), 3))
for ax, fk in zip(axs, MODES_KHZ):
    for spot, style in ((SPOT_UP, 'o-'), (SPOT_DOWN, 's--')):
        V, A = [], []
        for i, c in enumerate(conditions):
            if c.spot != spot: continue
            A.append(np.mean([abs(mode_peak(series['Z'][i, j], series['freq_Hz'], fk)[0])
                              for j in range(series['x_um'].size)])); V.append(c.bias_V)
        o = np.argsort(V); ax.plot(np.array(V)[o], np.array(A)[o], style, label=f'spot {spot}')
    ax.set_title(f'{fk:.0f} kHz', fontsize=9); ax.set_xlabel('DC bias (V)')
axs[0].set_ylabel('|Z| at mode peak, mean over positions'); axs[0].legend(frameon=False, fontsize=7)
plt.tight_layout(); plt.savefig(os.path.join(file_loc, 'bias_dependence_per_mode.png'), dpi=180); plt.show()
''')
ratio_md = md('''### Electrostatic / piezo ratio per mode

Piezo term flips with domain and is bias-independent: P = (Z_up − Z_down)/2 at each V.
Electrostatic term is domain-independent and linear in V: E(V) = (Z_up + Z_down)/2 = S·(V − V_cpd).
Fit S and V_cpd from E(V); the ratio that matters for interpretation is |S·V_cpd| / |P| — the
electrostatic share of the 0 V signal. Prediction: ≈ 4 at CR1, ≈ 0.6 at CR2, ≲ 0.1 at CR4–CR5.''')
ratio = code('''def two_domain_split(series, conditions, fk):
    """Per position: P (piezo, bias-averaged), S, V_cpd, and the 0 V electrostatic/piezo ratio."""
    f = series['freq_Hz']; rows = []
    for j in range(series['x_um'].size):
        zu = {c.bias_V: mode_peak(series['Z'][i, j], f, fk)[0] for i, c in enumerate(conditions) if c.spot == SPOT_UP}
        zd = {c.bias_V: mode_peak(series['Z'][i, j], f, fk)[0] for i, c in enumerate(conditions) if c.spot == SPOT_DOWN}
        V = np.array(sorted(set(zu) & set(zd)))
        P = np.array([(zu[v] - zd[v]) / 2 for v in V]); E = np.array([(zu[v] + zd[v]) / 2 for v in V])
        A = np.c_[V, np.ones_like(V)]; (S, c0), *_ = np.linalg.lstsq(A, E, rcond=None)
        v_cpd = -np.real(c0 / S) if abs(S) > 0 else np.nan
        Pm = P.mean(); resid = np.linalg.norm(E - (S * V + c0)) / np.linalg.norm(E)
        rows.append(dict(x=series['x_um'][j], P=abs(Pm), P_scatter=np.std(np.abs(P)) / abs(Pm),
                         S=abs(S), v_cpd=v_cpd, ratio0=abs(S * v_cpd) / abs(Pm), lin_resid=resid))
    return rows

print('mode    <ratio0>   V_cpd (mean +/- sd)   P scatter over V   E(V) linearity resid')
summary = {}
for fk in MODES_KHZ:
    r = two_domain_split(series, conditions, fk)
    ratio = np.array([q['ratio0'] for q in r]); vc = np.array([q['v_cpd'] for q in r])
    summary[fk] = dict(ratio0=np.median(ratio), v_cpd=np.nanmean(vc), rows=r)
    print(f'{fk:7.1f}  {np.median(ratio):8.2f}   {np.nanmean(vc):+6.2f} +/- {np.nanstd(vc):4.2f}     '
          f'{np.mean([q["P_scatter"] for q in r]):.2f}              {np.mean([q["lin_resid"] for q in r]):.2f}')
print('prediction from the wideband ladder: ratio0 ~ 4 (CR1), ~0.6 (CR2), <0.1 (CR4-CR5); V_cpd should be the same for every mode')
''')
null_md = md('''### CR1 null position vs bias (free-end cluster)

At 0 V the mode-1 null is 12.6 µm from the tip (x ≈ 432.4). If CR1 is electrostatic-dominated the
null is mostly the electrostatic null and should move as V_dc changes the mixture; CR3's null
positions (159.5, 299.1 µm — outside the cluster) are the piezo reference and should not.''')
null = code('''fc = cluster['freq_Hz']; xc = cluster['x_um']
def null_position(Zmap_x, x):
    """Parabolic minimum of log|Z| along x."""
    a = np.log(np.abs(Zmap_x) + 1e-15); i = int(np.argmin(a))
    if 0 < i < len(a) - 1:
        y0, y1, y2 = a[i - 1], a[i], a[i + 1]; den = y0 - 2 * y1 + y2
        return x[i] + (x[1] - x[0]) * 0.5 * (y0 - y2) / den if den != 0 else x[i]
    return x[i]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
res = []
for i, c in enumerate(conditions_c):
    prof = np.array([mode_peak(cluster['Z'][i, j], fc, MODES_KHZ[0])[0] for j in range(xc.size)])
    xn = null_position(prof, xc); res.append((c.bias_V, xn, prof))
    ax[0].semilogy(xc, np.abs(prof) / np.abs(prof).max(), 'o-', ms=3, label=f'{c.bias_V:+.0f} V')
res.sort()
ax[0].set_xlabel('x (um)'); ax[0].set_ylabel('|Z| at CR1 (norm.)'); ax[0].legend(fontsize=7, frameon=False)
ax[1].plot([r[0] for r in res], [PROBE_L_UM - r[1] for r in res], 'ko-')
ax[1].axhline(12.6, color='grey', ls=':', label='0 V dense sweep, 12.6 um'); ax[1].legend(frameon=False, fontsize=8)
ax[1].set_xlabel('DC bias (V)'); ax[1].set_ylabel('CR1 null, um from the free end')
plt.tight_layout(); plt.savefig(os.path.join(file_loc, 'cr1_null_vs_bias.png'), dpi=180); plt.show()
for V, xn, _ in res: print(f'{V:+.0f} V: CR1 null at x = {xn:.1f} um  ({PROBE_L_UM - xn:.1f} um from the free end)')
''')
cells[i + 1:i + 1] = [ratio_md, ratio, null_md, null]

json.dump(nb, open(p, 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
print('patched', p, len(cells), 'cells')
