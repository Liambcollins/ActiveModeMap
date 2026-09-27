"""Corrected-phase EB / EB+GP / low-rank benchmark on an SCM-PIT dense grid (A or B).

Differences from gridb.py (2026-09-17 audit, recommended order):
  * position axis from the package InvOLS ruler (`fit_position_scale` on the
    grid's own DenseReference_log.csv), not the manuscript's hard-coded line;
  * contact position either FITTED inside the node-fit band (218-224 um on the
    ruler axis) or PINNED to the node fit (221.9 um) with --pin;
  * study-compatible settings for Grid_A: band 270-450 kHz, FDEC 4, n = 8.

usage: python3 gridfit.py A|B corrected|raw N1,N2,... [--pin]
"""
import sys, time, json
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from activemodemap.forward_model import EBForwardModel, ProbeGeometry
from activemodemap.inference import PhysicsPosterior
from activemodemap.hybrid import HybridSurrogate
from activemodemap.lowrank import reconstruct_map, dns_from_map, spatial_null
from activemodemap.asylum import phase_slope_through_peak
from activemodemap.online import fit_position_scale

U = '/mnt/user-data/uploads/User Data--Liam--ActiveModeMap/SCM_PIT/'
GRIDS = dict(
    A=dict(dir='Dense_Grid_A', band=(270e3, 450e3), fdec=4, n_sparse=20),
    B=dict(dir='Dense_Grid_B', band=(250e3, 400e3), fdec=2, n_sparse=None),
)
L_UM = 232.0                 # operator's 225 anchor is 3-9 um short of the free end
K_LEVER = 2.387
F0_GUESS = 60.0e3
NODE_CONTACT = 221.9         # tip-position-fit-2026-08-14 (Grid_B, ruler axis)
CONTACT_BAND = (218.0, 224.0)


def position_scale(g):
    dl = pd.read_csv(U + g['dir'] + '/DenseReference_log.csv')
    dl['i'] = dl.tune_file.str.extract(r'_(\d{4})\.txt$')[0].astype(int)
    nsp = g['n_sparse']
    if nsp is None:                       # first run of 1-um steps starts the dense pass
        d = np.abs(np.diff(dl.position_x_um.values)); nsp = int(np.argmax(d <= 1.01))
        while nsp < len(d) - 3 and not np.all(d[nsp:nsp + 3] <= 1.01): nsp += 1
    return fit_position_scale(dl[dl.i < nsp], dl[dl.i >= nsp], verbose=False)


def load(g):
    d = np.load(U + g['dir'] + '/DenseReference.npz', allow_pickle=True)
    ps = position_scale(g)
    x = ps.apply(d['x_um'].astype(float)); f = d['freq_Hz']; Z = d['Z']
    m = (f >= g['band'][0]) & (f <= g['band'][1])
    f, Z = f[m][::g['fdec']], Z[:, m][:, ::g['fdec']]
    o = np.argsort(x)
    return x[o], f, Z[o], ps


def build_model(f_Hz, f0_hz=F0_GUESS, setback=L_UM - NODE_CONTACT):
    geom = ProbeGeometry(name='SCM-PIT', f0_hz=f0_hz, k_lever=K_LEVER, L_um=L_UM,
                         tip_setback_um=setback, tip_height_um=12.5, tilt_deg=11.0,
                         tip_mass_ratio=0.004)
    fs = f0_hz / 1.87510407 ** 2
    m = EBForwardModel(geom=geom, n_modes=12, nx=481, nf=f_Hz.size,
                       omega_lo=float(f_Hz[0] / fs), omega_hi=float(f_Hz[-1] / fs))
    m.PRIOR_LO = np.array([1.0, 1.0, 0.5, -4.0, -4.0]); m.PRIOR_HI = np.array([5.0, 4.0, 2.5, 1.0, 1.0])
    return m


def sigma_est(Z, half=6):
    from scipy.ndimage import uniform_filter1d
    r = Z - (uniform_filter1d(Z.real, 2 * half + 1, axis=1) + 1j * uniform_filter1d(Z.imag, 2 * half + 1, axis=1))
    a = np.abs(r).ravel(); return float(np.median(a) + 1.4826 * np.median(np.abs(a - np.median(a))))


def dns(x_um, Zmap, f, guess=None):
    A = np.abs(Zmap); ires = int(np.argmax(A[len(x_um) // 2]))
    try:
        v = dns_from_map(x_um, Zmap, f, ires, guess_um=guess)
    except Exception:
        v = None
    if v is None or not np.isfinite(v):
        v = spatial_null(x_um, Zmap, f, ires, guess_um=guess)
    return float(v) if np.isfinite(v) else np.nan


def signchange_null(x_um, Zmap, f, guess):
    A = np.abs(Zmap); ires = int(np.argmax(A[len(x_um) // 2]))
    v = spatial_null(x_um, Zmap, f, ires, guess_um=guess)
    return float(v) if np.isfinite(v) else np.nan


def run(grid='A', N_list=(8,), conj=True, pin=False, tag=None, verbose=True):
    g = GRIDS[grid]
    tag = tag or f"{grid}_{'corrected' if conj else 'raw'}{'_pin' if pin else ''}"
    x, f, Zraw, ps = load(g)
    Z = np.conj(Zraw) if conj else Zraw
    scale = np.abs(Z).max(); Zn = Z / scale
    sig = sigma_est(Zn)
    m0 = build_model(f)
    xi = x / L_UM
    truth = dns(x, Zn, f); truth_sc = signchange_null(x, Zn, f, guess=truth)
    slope = np.median([phase_slope_through_peak(f, np.abs(z), np.degrees(np.angle(z))) for z in Zn[::max(1, x.size // 10)]])
    if verbose:
        print(f"[{tag}] axis {ps}; {x.size} pos, x {x[0]:.1f}-{x[-1]:.1f} um, band {f[0]/1e3:.0f}-{f[-1]/1e3:.0f} kHz "
              f"({f.size} pts); phase through peak {slope:+.0f} deg; truth D-NS branch {truth:.2f} / sign {truth_sc:.2f} um; sigma/max {sig:.1e}")
    rows = []
    for N in N_list:
        idx = np.unique(np.round(np.linspace(0, x.size - 1, N)).astype(int))
        held = np.setdiff1d(np.arange(x.size), idx)
        data = [{'x': xi[i], 'plus': Zn[i]} for i in idx]
        t0 = time.time()
        if pin:
            post = PhysicsPosterior(m0, sigma=sig, rng=np.random.default_rng(0), fit_zeta=True,
                                    fit_geometry='f0', analytic_gain=True, f0_bounds_hz=(50e3, 100e3))
        else:
            post = PhysicsPosterior(m0, sigma=sig, rng=np.random.default_rng(0), fit_zeta=True,
                                    fit_geometry=True, analytic_gain=True, f0_bounds_hz=(50e3, 100e3),
                                    setback_bounds_um=(L_UM - CONTACT_BAND[1], L_UM - CONTACT_BAND[0]))
        th = post.fit(data)
        cols = [int(np.argmin(np.abs(post.model.xi - v))) for v in xi]
        eb = post.predict_map()[:, cols].T
        hy = HybridSurrogate(post).update(data).corrected_maps()[0][:, cols].T
        rank = max(2, min(4, len(idx) - 1))
        lr = reconstruct_map(x, idx, Zn[idx], rank=rank)['Zrec']
        err = lambda M: 100 * np.linalg.norm(M[held] - Zn[held]) / np.linalg.norm(Zn[held])
        aerr = lambda M: 100 * np.linalg.norm(np.abs(M[held]) - np.abs(Zn[held])) / np.linalg.norm(np.abs(Zn[held]))
        wgt = np.abs(Zn[held]) > 0.1 * np.abs(Zn).max()
        pmae = lambda M: float(np.degrees(np.mean(np.abs(np.angle(M[held][wgt] * np.conj(Zn[held][wgt]))))))
        d = dict(zip(post.names, th))
        contact = L_UM - d.get('tip_setback_um', L_UM - NODE_CONTACT)
        row = dict(grid=grid, N=len(idx), rank=rank, pin=pin, conj=conj,
                   err_eb=err(eb), err_hy=err(hy), err_lr=err(lr),
                   aerr_eb=aerr(eb), aerr_hy=aerr(hy), aerr_lr=aerr(lr),
                   pmae_eb=pmae(eb), pmae_hy=pmae(hy), pmae_lr=pmae(lr),
                   dns_eb=dns(x, eb, f, truth), dns_hy=dns(x, hy, f, truth), dns_lr=dns(x, lr, f, truth),
                   sc_eb=signchange_null(x, eb, f, truth), sc_hy=signchange_null(x, hy, f, truth), sc_lr=signchange_null(x, lr, f, truth),
                   theta={k: float(v) for k, v in d.items()}, gain=[float(abs(post.gain)), float(np.degrees(np.angle(post.gain)))],
                   contact_um=contact, red_chi2=float(post.red_chi2), dt=time.time() - t0,
                   truth=truth, truth_sc=truth_sc, axis=str(ps))
        rows.append(row)
        with open(f'grid_{tag}_rows.jsonl', 'a') as fh: fh.write(json.dumps(row) + '\n')
        if verbose:
            print(f"[{tag}] N={row['N']:2d} | complex err  EB {row['err_eb']:6.1f}%  EB+GP {row['err_hy']:5.1f}%  LR {row['err_lr']:5.1f}% | "
                  f"amp err EB {row['aerr_eb']:5.1f} EB+GP {row['aerr_hy']:5.1f} LR {row['aerr_lr']:5.1f} | "
                  f"phase MAE EB {row['pmae_eb']:4.1f} EB+GP {row['pmae_hy']:4.1f} LR {row['pmae_lr']:4.1f} deg | "
                  f"D-NS(branch) err EB {row['dns_eb']-truth:+5.2f} EB+GP {row['dns_hy']-truth:+5.2f} LR {row['dns_lr']-truth:+5.2f} | "
                  f"sign EB {row['sc_eb']-truth:+5.2f} EB+GP {row['sc_hy']-truth:+5.2f} LR {row['sc_lr']-truth:+5.2f} | "
                  f"f0 {d['f0_kHz']:.1f} kHz contact {contact:.1f} um zeta {10**d['log_zeta']:.4f} k*={10**d['log_alpha']*K_LEVER:.0f} N/m ({row['dt']:.0f}s)")
    return rows


if __name__ == '__main__':
    grid = sys.argv[1]; mode = sys.argv[2]
    Ns = tuple(int(v) for v in sys.argv[3].split(','))
    run(grid, Ns, conj=(mode == 'corrected'), pin=('--pin' in sys.argv))
