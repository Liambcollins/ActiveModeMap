"""Step 3: the manuscript's EB+GP arm on SCM-PIT Dense_Grid_B with the corrected fit.

Mirrors the 2026-08-15 benchmark protocol: mode-A band 250-400 kHz, equidistant
designs, error on HELD-OUT positions, InvOLS-corrected axis
x_true = 0.843 x_cmd + 37.5 um (manuscript; 0.5 um rms). Truth D-NS recomputed
here with the package estimator on that axis.

Arms: EB alone / EB+GP (package PhysicsPosterior, fit_zeta + fit_geometry +
analytic_gain) on the phase-corrected data, the same on the RAW-convention data
(the ablation that should reproduce the ~142% "phase defect"), and low-rank.
"""
import sys, time, json
import numpy as np
sys.path.insert(0, '.')
from activemodemap.forward_model import EBForwardModel, ProbeGeometry
from activemodemap.inference import PhysicsPosterior
from activemodemap.hybrid import HybridSurrogate
from activemodemap.lowrank import reconstruct_map, dns_from_map, spatial_null
from activemodemap.asylum import phase_slope_through_peak

NPZ = '/mnt/user-data/uploads/User Data--Liam--ActiveModeMap/SCM_PIT/Dense_Grid_B/DenseReference.npz'
BAND = (250e3, 400e3)
L_UM = 232.0                 # free end unknown; doc: operator's 225 anchor is 3-9 um short
K_LEVER = 2.387
F0_GUESS = 58.2e3            # implied free f0 from the 2026-08-15 fits; fitted here
FDEC = 2                     # 109 Hz -> 219 Hz step in the band (Q~205 at 293 kHz: FWHM 1.4 kHz -> 6.5 pts)


def load():
    d = np.load(NPZ)
    x_cmd = d['x_um']; f = d['freq_Hz']; Z = d['Z']
    x = 0.843 * x_cmd + 37.5                     # InvOLS-corrected axis
    m = (f >= BAND[0]) & (f <= BAND[1])
    f, Z = f[m][::FDEC], Z[:, m][:, ::FDEC]
    o = np.argsort(x)
    return x[o], f, Z[o]


def build_model(f_Hz, f0_hz=F0_GUESS, setback=L_UM - 221.9):
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


def dns(x_um, Zmap, f, L, guess=None):
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


def run(N_list=(3, 4, 6, 12, 20), conj=True, tag='corrected', verbose=True):
    x, f, Zraw = load()
    Z = np.conj(Zraw) if conj else Zraw
    scale = np.abs(Z).max(); Zn = Z / scale
    sig = sigma_est(Zn)
    m0 = build_model(f)
    xi = x / L_UM
    truth = dns(x, Zn, f, L_UM)
    truth_sc = signchange_null(x, Zn, f, guess=truth)
    slope = np.median([phase_slope_through_peak(f, np.abs(z), np.degrees(np.angle(z))) for z in Zn[::24]])
    if verbose:
        print(f"[{tag}] {x.size} pos, x {x[0]:.1f}-{x[-1]:.1f} um, band {f[0]/1e3:.0f}-{f[-1]/1e3:.0f} kHz "
              f"({f.size} pts); phase through peak {slope:+.0f} deg; truth D-NS branch {truth:.2f} / sign {truth_sc:.2f} um; sigma/max {sig:.1e}")
    rows = []
    for N in N_list:
        idx = np.unique(np.round(np.linspace(0, x.size - 1, N)).astype(int))
        held = np.setdiff1d(np.arange(x.size), idx)
        data = [{'x': xi[i], 'plus': Zn[i]} for i in idx]
        t0 = time.time()
        post = PhysicsPosterior(m0, sigma=sig, rng=np.random.default_rng(0), fit_zeta=True,
                                fit_geometry=True, analytic_gain=True,
                                f0_bounds_hz=(50e3, 100e3), setback_bounds_um=(4.0, 14.0))
        th = post.fit(data)
        cols = [int(np.argmin(np.abs(post.model.xi - v))) for v in xi]
        eb = post.predict_map()[:, cols].T
        hy = HybridSurrogate(post).update(data).corrected_maps()[0][:, cols].T
        rank = max(2, min(4, len(idx) - 1))
        lr = reconstruct_map(x, idx, Zn[idx], rank=rank)['Zrec']
        err = lambda M: 100 * np.linalg.norm(M[held] - Zn[held]) / np.linalg.norm(Zn[held])
        aerr = lambda M: 100 * np.linalg.norm(np.abs(M[held]) - np.abs(Zn[held])) / np.linalg.norm(np.abs(Zn[held]))
        # phase MAE where the signal is significant
        wgt = np.abs(Zn[held]) > 0.1 * np.abs(Zn).max()
        pmae = lambda M: float(np.degrees(np.mean(np.abs(np.angle(M[held][wgt] * np.conj(Zn[held][wgt]))))))
        d = dict(zip(post.names, th))
        row = dict(N=len(idx), rank=rank,
                   err_eb=err(eb), err_hy=err(hy), err_lr=err(lr),
                   aerr_eb=aerr(eb), pmae_eb=pmae(eb), pmae_hy=pmae(hy), pmae_lr=pmae(lr),
                   dns_eb=dns(x, eb, f, L_UM, truth), dns_hy=dns(x, hy, f, L_UM, truth), dns_lr=dns(x, lr, f, L_UM, truth),
                   sc_eb=signchange_null(x, eb, f, truth), sc_hy=signchange_null(x, hy, f, truth), sc_lr=signchange_null(x, lr, f, truth),
                   theta={k: float(v) for k, v in d.items()}, gain=[float(abs(post.gain)), float(np.degrees(np.angle(post.gain)))],
                   contact_um=L_UM - d['tip_setback_um'], red_chi2=float(post.red_chi2), dt=time.time() - t0)
        rows.append(row)
        with open(f'gridb_{tag}_rows.jsonl','a') as fh: fh.write(json.dumps(dict(row, truth=truth, truth_sc=truth_sc))+'\n')
        if verbose:
            print(f"[{tag}] N={row['N']:2d} | mode-A complex err  EB {row['err_eb']:6.1f}%  EB+GP {row['err_hy']:5.1f}%  LR {row['err_lr']:5.1f}% | "
                  f"EB amp {row['aerr_eb']:5.1f}%  phase MAE EB {row['pmae_eb']:4.1f} EB+GP {row['pmae_hy']:4.1f} LR {row['pmae_lr']:4.1f} deg | "
                  f"D-NS(branch) err  EB {row['dns_eb']-truth:+5.2f} EB+GP {row['dns_hy']-truth:+5.2f} LR {row['dns_lr']-truth:+5.2f} | "
                  f"sign-change EB {row['sc_eb']-truth:+5.2f} EB+GP {row['sc_hy']-truth:+5.2f} LR {row['sc_lr']-truth:+5.2f} | "
                  f"f0 {d['f0_kHz']:.1f} kHz contact {row['contact_um']:.1f} um zeta {10**d['log_zeta']:.4f} k*={10**d['log_alpha']*K_LEVER:.0f} N/m ({row['dt']:.0f}s)")
    return dict(rows=rows, truth=truth, truth_sc=truth_sc, slope=slope, x=x.tolist())


if __name__ == '__main__':
    tag = sys.argv[1] if len(sys.argv) > 1 else 'corrected'
    Ns = tuple(int(v) for v in sys.argv[2].split(',')) if len(sys.argv) > 2 else (3, 4, 6, 12, 20)
    out = run(N_list=Ns, conj=(tag == 'corrected'), tag=tag)
    json.dump(out, open(f'gridb_{tag}.json', 'w'))
