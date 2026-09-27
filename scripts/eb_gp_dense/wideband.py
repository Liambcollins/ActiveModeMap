"""Wideband (5-995 kHz) recovery on the PPP-CONTAu dense set: what happens to the
higher contact modes, their nodes and antinodes, when the arms are trained on a
few positions.

Arms (all leakage-safe: see only the revealed rows)
  EB-wide      PhysicsPosterior fitted on the whole band (zeta, setback, gain free)
  EB+GP-wide   the same plus the discrepancy GP on the wideband residual
  EB-band1     PhysicsPosterior fitted on 50-80 kHz only, theta extrapolated to the
               whole band through the modal model (the study's "library covers 25 %"
               situation, but with a model that CAN predict the higher modes)
  LR-4 / LR-h  Chebyshev low-rank, rank 4 and rank min(N-1, 8)

Feature metrics per contact mode (windows around the measured peaks): complex
NRMSE on held-out positions, median |dB| amplitude error on signal bins, peak
frequency shift, antinode position, node positions and node depth.

usage: python3 wideband.py N1,N2,... [--fdec 5]
"""
import sys, time, json
import numpy as np
sys.path.insert(0, '.')
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
from activemodemap.forward_model import EBForwardModel, ProbeGeometry
from activemodemap.inference import PhysicsPosterior
from activemodemap.hybrid import HybridSurrogate
from activemodemap.lowrank import reconstruct_map

NPZ = '/mnt/user-data/uploads/091726 Softprobe/DenseSweep_PPPCONTAU_100nN.npz'
F0_HZ = 13.649e3; K_LEVER = 0.4057; L_UM = 445.0
OM1 = 1.87510407 ** 2
MODES_KHZ = [63.4, 200.1, 408.1, 667.3, 916.0]        # contact flexural modes (measured)
EXTRA_KHZ = [281.9, 627.1]                             # weak non-flexural features
HALF_WIN = dict(default=4e3)                            # +-4 kHz windows (scaled by f/400 kHz above that)


DATASETS = dict(
    ppp=dict(npz=NPZ, f0=13.649e3, k=0.4057, L=445.0, tip_mass=0.0018, setback=12.2, sb_bounds=(4.0, 20.0),
             modes=[63.4, 200.1, 408.1, 667.3, 916.0], extra=[281.9, 627.1], floor_band=(950e3, 1e9), axis=None, band1=(50e3, 80e3)),
    scmpit=dict(npz='/mnt/user-data/uploads/User Data--Liam--ActiveModeMap/SCM_PIT/Dense_Grid_B/DenseReference.npz',
                f0=60.0e3, k=2.387, L=232.0, tip_mass=0.004, setback=10.1, sb_bounds=(8.0, 14.0),
                modes=[293.2, 904.2], extra=[978.2, 734.0], floor_band=(1.5e6, 1.7e6), axis=(0.8319, 38.69), band1=(250e3, 400e3)),
)
DS = 'ppp'


def configure(name):
    """Point the module at one dataset (sets the geometry globals)."""
    global DS, NPZ, F0_HZ, K_LEVER, L_UM, MODES_KHZ, EXTRA_KHZ
    c = DATASETS[name]; DS = name
    NPZ, F0_HZ, K_LEVER, L_UM = c['npz'], c['f0'], c['k'], c['L']
    MODES_KHZ, EXTRA_KHZ = c['modes'], c['extra']


def load(fdec=5):
    c = DATASETS[DS]
    d = np.load(NPZ, allow_pickle=True)
    x = d['x_um'].astype(float)
    f = d['freq'] if 'freq' in d.files else d['freq_Hz']; f = f[0] if f.ndim == 2 else f
    Z = np.conj(d['Z'])                                  # Igor-raw -> model convention
    if c['axis']: x = c['axis'][0] * x + c['axis'][1]   # InvOLS-ruler axis
    o = np.argsort(x); x, Z = x[o], Z[o]
    f, Z = f[::fdec], Z[:, ::fdec]
    scale = np.abs(Z).max()
    return x, f, Z / scale


def geom(setback=None):
    c = DATASETS[DS]
    return ProbeGeometry(name=DS, f0_hz=F0_HZ, k_lever=K_LEVER, L_um=L_UM,
                         tip_setback_um=c['setback'] if setback is None else setback,
                         tip_height_um=12.5, tilt_deg=11.0, tip_mass_ratio=c['tip_mass'])


def model_on(f_Hz, g=None, nx=241):
    fs = F0_HZ / OM1
    m = EBForwardModel(geom=g or geom(), n_modes=12, nx=nx, nf=f_Hz.size,
                       omega_lo=float(f_Hz[0] / fs), omega_hi=float(f_Hz[-1] / fs))
    m.PRIOR_LO = np.array([1.0, 1.0, 0.5, -4.0, -4.0]); m.PRIOR_HI = np.array([5.0, 4.0, 2.5, 1.0, 1.0])
    if FREQ_FIRST:      # mode-frequency-first: alpha, kcone from the five measured peaks
        m.PRIOR_LO[:2] = [FREQ_FIRST[0] - 0.01, FREQ_FIRST[1] - 0.01]
        m.PRIOR_HI[:2] = [FREQ_FIRST[0] + 0.01, FREQ_FIRST[1] + 0.01]
    return m


#: (log_alpha, log_kcone, setback_um) that place the five contact modes within
#: 0.7 % rms of the measured 63.47/200.12/408.02/667.34/916.14 kHz (grid search
#: over the model box; see the 2026-09-17 wideband doc).  None = free fit.
FREQ_FIRST = None
TAG = 'wideband'


def sigma_est(Z, half=6):
    r = Z - (uniform_filter1d(Z.real, 2 * half + 1, axis=1) + 1j * uniform_filter1d(Z.imag, 2 * half + 1, axis=1))
    a = np.abs(r).ravel(); return float(np.median(a) + 1.4826 * np.median(np.abs(a - np.median(a))))


def fit_arm(x, f, Zn, idx, band=None):
    """PhysicsPosterior on the rows idx; band=(lo,hi) restricts the FIT grid.
    Returns (post, predict_on(f_grid) -> (nf, npos) complex map at the data x)."""
    xi = x / L_UM
    if band is None:
        fm, Zm = f, Zn
    else:
        w = (f >= band[0]) & (f <= band[1]); fm, Zm = f[w], Zn[:, w]
    m0 = model_on(fm)
    data = [{'x': xi[i], 'plus': Zm[i]} for i in idx]
    sb = (FREQ_FIRST[2] - 0.1, FREQ_FIRST[2] + 0.1) if FREQ_FIRST else DATASETS[DS]['sb_bounds']
    post = PhysicsPosterior(m0, sigma=sigma_est(Zm), rng=np.random.default_rng(0), fit_zeta=True,
                            fit_geometry='setback', analytic_gain=True, setback_bounds_um=sb)
    th = post.fit(data)

    def predict_on(f_grid):
        mw = model_on(f_grid, g=post.model.geom)
        resp = mw.response(post._core(th))
        cols = [int(np.argmin(np.abs(mw.xi - v))) for v in xi]
        return (post.gain * (resp['piezo'] + resp['eps'] * resp['elec']))[:, cols]
    return post, th, data, predict_on


def parabolic_min(xv, yv):
    i = int(np.argmin(yv))
    if 0 < i < len(yv) - 1:
        y0, y1, y2 = yv[i - 1], yv[i], yv[i + 1]
        den = (y0 - 2 * y1 + y2)
        return xv[i] + (xv[1] - xv[0]) * 0.5 * (y0 - y2) / den if den != 0 else xv[i]
    return xv[i]


def nodes_along_x(x, col, rel=0.25):
    """Local minima of |Z| along x that are nodes: the profile rises by at least
    1/rel on one side and by at least 2x on the other (so the tip-end fall-off
    of CR1 and a node sitting at the first/last position both count, while the
    smooth roll-off toward a span edge does not)."""
    a = np.log(np.abs(col) + 1e-15)
    out = []
    n = len(a)
    for i in range(n):
        lo, hi = max(i - 1, 0), min(i + 1, n - 1)
        if not (a[i] <= a[lo] and a[i] <= a[hi]):
            continue
        left = a[:i].max() - a[i] if i > 0 else 0.0
        right = a[i + 1:].max() - a[i] if i < n - 1 else 0.0
        if max(left, right) >= np.log(1 / rel) and min(left, right) >= np.log(2.0) or \
           (i in (0, 1, n - 2, n - 1) and max(left, right) >= np.log(1 / rel)):
            out.append(parabolic_min(x[max(i - 1, 0):i + 2], a[max(i - 1, 0):i + 2]) if 0 < i < n - 1 else float(x[i]))
    # merge minima closer than two samples
    out = sorted(out); merged = []
    for v in out:
        if merged and abs(v - merged[-1]) < 2.5 * abs(x[1] - x[0]): continue
        merged.append(float(v))
    return merged


def feature_metrics(Zrec, Ztrue, f, x, held, floor):
    out = {}
    A_t, A_r = np.abs(Ztrue), np.abs(Zrec)
    sig = A_t > 3 * floor
    db = 20 * np.log10((A_r + 1e-15) / (A_t + 1e-15))
    out['wide'] = dict(
        cplx_nrmse=100 * np.linalg.norm(Zrec[held] - Ztrue[held]) / np.linalg.norm(Ztrue[held]),
        med_abs_db=float(np.median(np.abs(db[held][sig[held]]))),
        within3db=float(np.mean(np.abs(db[held][sig[held]]) < 3)),
        spurious=float(np.mean((A_r[held] > 3 * floor) & (A_t[held] < floor))),
        phase_mae=float(np.degrees(np.mean(np.abs(np.angle(Zrec[held][sig[held]] * np.conj(Ztrue[held][sig[held]])))))))
    for fk in MODES_KHZ + EXTRA_KHZ:
        w = np.abs(f - fk * 1e3) <= HALF_WIN['default'] * max(1.0, fk / 400.0)
        if w.sum() < 4: continue
        Zt, Zr = Ztrue[:, w], Zrec[:, w]
        At, Ar = np.abs(Zt), np.abs(Zr)
        s = At[held] > 3 * floor
        # peak frequency per held-out position (only where the mode is visible)
        vis = At[held].max(1) > 5 * floor
        fpk_t = f[w][np.argmax(At[held], 1)]; fpk_r = f[w][np.argmax(Ar[held], 1)]
        # x-profile at the measured peak frequency of the mode
        jt = int(np.argmax(At.mean(0)))
        prof_t, prof_r = At[:, jt], Ar[:, jt]
        nt, nr = nodes_along_x(x, prof_t), nodes_along_x(x, prof_r)
        node_err = [float(min(abs(n - m) for m in nr)) if nr else np.nan for n in nt]
        depth_t = prof_t.max() / prof_t.min(); depth_r = prof_r.max() / prof_r.min()
        dbw = 20 * np.log10((Ar[held] + 1e-15) / (At[held] + 1e-15))
        out[f'{fk:.1f}'] = dict(
            cplx_nrmse=100 * np.linalg.norm(Zr[held] - Zt[held]) / np.linalg.norm(Zt[held]),
            med_abs_db=float(np.median(np.abs(dbw[s]))) if s.any() else np.nan,
            peak_amp_db=float(20 * np.log10(Ar.max() / At.max())),
            fpk_shift_hz=float(np.median(fpk_r[vis] - fpk_t[vis])) if vis.any() else np.nan,
            fpk_spread_hz=float(np.median(np.abs(fpk_r[vis] - fpk_t[vis]))) if vis.any() else np.nan,
            antinode_true=float(x[np.argmax(prof_t)]), antinode_rec=float(x[np.argmax(prof_r)]),
            nodes_true=[round(v, 1) for v in nt], nodes_rec=[round(v, 1) for v in nr],
            node_err_um=[round(v, 2) for v in node_err],
            depth_true=float(depth_t), depth_rec=float(depth_r))
    return out


def run(N_list, fdec=5):
    x, f, Zn = load(fdec)
    fb = DATASETS[DS]['floor_band']; floor = float(np.median(np.abs(Zn[:, (f > fb[0]) & (f < fb[1])])))
    print(f"{x.size} pos x {f.size} bins ({f[1]-f[0]:.0f} Hz), floor/max {floor:.1e}")
    for N in N_list:
        idx = np.unique(np.round(np.linspace(0, x.size - 1, N)).astype(int))
        held = np.setdiff1d(np.arange(x.size), idx)
        res = dict(N=int(len(idx)), idx=idx.tolist(), arms={})
        t0 = time.time()
        post, th, data, pred_wide = fit_arm(x, f, Zn, idx)
        eb = pred_wide(f).T
        hy = HybridSurrogate(post).update(data).corrected_maps()[0]
        cols = [int(np.argmin(np.abs(post.model.xi - v))) for v in x / L_UM]
        hy = hy[:, cols].T
        d = dict(zip(post.names, th))
        res['theta_wide'] = {k: float(v) for k, v in d.items()}
        print(f"N={len(idx)} EB-wide fit {time.time()-t0:.0f}s: setback {d['tip_setback_um']:.1f} zeta {10**d['log_zeta']:.4f} "
              f"Q {10**d['log_Q']:.0f} k*={10**d['log_alpha']*K_LEVER:.1f} eps {10**d['log_eps']:.2f} chi2 {post.red_chi2:.1f}")
        t1 = time.time()
        postb, thb, _, pred_b1 = fit_arm(x, f, Zn, idx, band=DATASETS[DS]['band1'])
        ebb = pred_b1(f).T
        db_ = dict(zip(postb.names, thb))
        res['theta_band1'] = {k: float(v) for k, v in db_.items()}
        print(f"      EB-band1 fit {time.time()-t1:.0f}s: setback {db_['tip_setback_um']:.1f} zeta {10**db_['log_zeta']:.4f} Q {10**db_['log_Q']:.0f} k*={10**db_['log_alpha']*K_LEVER:.1f}")
        lr4 = reconstruct_map(x, idx, Zn[idx], rank=max(2, min(4, len(idx) - 1)))['Zrec']
        rh = max(2, min(8, len(idx) - 1))
        lrh = reconstruct_map(x, idx, Zn[idx], rank=rh)['Zrec']
        for name, M in (('EB-wide', eb), ('EB+GP-wide', hy), ('EB-band1', ebb), ('LR-4', lr4), (f'LR-{rh}', lrh)):
            res['arms'][name] = feature_metrics(M, Zn, f, x, held, floor)
            w = res['arms'][name]['wide']
            print(f"   {name:11s} wide: cplx {w['cplx_nrmse']:6.1f}%  med|dB| {w['med_abs_db']:4.2f}  <3dB {100*w['within3db']:4.1f}%  spurious {100*w['spurious']:4.1f}%  phase {w['phase_mae']:4.1f} deg")
            for fk in MODES_KHZ:
                r = res['arms'][name].get(f'{fk:.1f}')
                if r: print(f"        {fk:6.1f} kHz: cplx {r['cplx_nrmse']:6.1f}%  med|dB| {r['med_abs_db']:5.2f}  peak {r['peak_amp_db']:+5.1f} dB  fpk {r['fpk_shift_hz']:+6.0f}/{r['fpk_spread_hz']:5.0f} Hz  antinode {r['antinode_true']:.0f}->{r['antinode_rec']:.0f}  nodes {r['nodes_true']}->{r['nodes_rec']} err {r['node_err_um']}  depth {r['depth_true']:.0f}->{r['depth_rec']:.0f}")
        res['freq_first'] = FREQ_FIRST
        with open(TAG + '_rows.jsonl', 'a') as fh: fh.write(json.dumps(res) + '\n')
        np.savez(f'{TAG}_maps_N{len(idx)}.npz', x=x, f=f, Z=Zn, eb=eb, hy=hy, ebb=ebb, lr4=lr4, lrh=lrh, idx=idx)


if __name__ == '__main__':
    Ns = tuple(int(v) for v in sys.argv[1].split(','))
    fdec = int(sys.argv[sys.argv.index('--fdec') + 1]) if '--fdec' in sys.argv else 5
    if '--data' in sys.argv:
        configure(sys.argv[sys.argv.index('--data') + 1]); TAG = 'wideband_' + DS
    if '--freq-first' in sys.argv:
        FREQ_FIRST = (3.1, 2.0, 10.0); TAG += '_ff'
    run(Ns, fdec)
