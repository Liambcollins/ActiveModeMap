"""Package-backed physics arm: the study's EB / EB+GP reconstruction through
`activemodemap.inference.PhysicsPosterior`, so there is ONE physics fitter in
the repository instead of the study's library-grid fit plus the package's.

Leakage-safe like every arm in cplxrec: sees only (x_grid, sel_idx, Z_sel,
freq).  Returns the same dict shape as `cplxrec.rec_phys_cplx` so
`run_recon_cplx.py` / `sweep_n_cplx.py` can call it as a drop-in arm.

Requires Z in the MODEL phase convention (setup_data.load does that) and the
2026-09-17 package: fit_zeta / fit_geometry / analytic_gain on PhysicsPosterior.

    from pkgrec import rec_eb_pkg
    out = rec_eb_pkg(x, sel, Z[sel], f, gp=True)          # Zrec (npos, nfreq)

Geometry defaults are the SCM-PIT/Multi75G lever of Dense_Grid_A/B on the
InvOLS-calibrated axis: L = 232 um (the operator's 225 anchor is 3-9 um short
of the free end), contact 221.9 um from the node fit (fitted inside 218-224
unless `pin_contact=True`), k_lever 2.387 N/m, f0 free in 50-100 kHz.
"""
import sys as _sys, os as _os
_D = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_D, 'config.py')):
    _D = _os.path.dirname(_D)
_sys.path[:0] = [_D, _os.path.join(_D, 'src')]

import numpy as np
from activemodemap.forward_model import EBForwardModel, ProbeGeometry
from activemodemap.inference import PhysicsPosterior
from activemodemap.hybrid import HybridSurrogate

L_UM = 232.0
K_LEVER = 2.387
NODE_CONTACT_UM = 221.9
CONTACT_BAND_UM = (218.0, 224.0)
F0_BOUNDS_HZ = (50e3, 100e3)


def build_model(freq, f0_hz=60e3, contact_um=NODE_CONTACT_UM, L_um=L_UM,
                k_lever=K_LEVER, n_modes=12, nx=481):
    geom = ProbeGeometry(name='SCM-PIT', f0_hz=f0_hz, k_lever=k_lever, L_um=L_um,
                         tip_setback_um=L_um - contact_um, tip_height_um=12.5,
                         tilt_deg=11.0, tip_mass_ratio=0.004)
    fs = f0_hz / 1.87510407 ** 2
    m = EBForwardModel(geom=geom, n_modes=n_modes, nx=nx, nf=len(freq),
                       omega_lo=float(freq[0] / fs), omega_hi=float(freq[-1] / fs))
    m.PRIOR_LO = np.array([1.0, 1.0, 0.5, -4.0, -4.0])
    m.PRIOR_HI = np.array([5.0, 4.0, 2.5, 1.0, 1.0])
    return m


def _sigma(Z, half=6):
    from scipy.ndimage import uniform_filter1d
    sm = uniform_filter1d(Z.real, 2 * half + 1, axis=1) + 1j * uniform_filter1d(Z.imag, 2 * half + 1, axis=1)
    a = np.abs(Z - sm).ravel()
    return float(np.median(a) + 1.4826 * np.median(np.abs(a - np.median(a))))


def rec_eb_pkg(x_grid, sel_idx, Z_sel, freq, gp=True, pin_contact=False,
               L_um=L_UM, seed=0, **kw):
    sel = np.asarray(sel_idx)
    x_grid = np.asarray(x_grid, float)
    scale = float(np.abs(Z_sel).max())
    Zs = np.asarray(Z_sel) / scale
    model = build_model(freq, L_um=L_um)
    xi = x_grid / L_um
    data = [{'x': xi[i], 'plus': Zs[k]} for k, i in enumerate(sel)]
    if pin_contact:
        post = PhysicsPosterior(model, sigma=_sigma(Zs), rng=np.random.default_rng(seed),
                                fit_zeta=True, fit_geometry='f0', analytic_gain=True,
                                f0_bounds_hz=F0_BOUNDS_HZ)
    else:
        post = PhysicsPosterior(model, sigma=_sigma(Zs), rng=np.random.default_rng(seed),
                                fit_zeta=True, fit_geometry=True, analytic_gain=True,
                                f0_bounds_hz=F0_BOUNDS_HZ,
                                setback_bounds_um=(L_um - CONTACT_BAND_UM[1], L_um - CONTACT_BAND_UM[0]))
    th = post.fit(data)
    cols = [int(np.argmin(np.abs(post.model.xi - v))) for v in xi]
    eb = post.predict_map()[:, cols].T
    if gp:
        sur = HybridSurrogate(post).update(data)
        Zrec = sur.corrected_maps()[0][:, cols].T
    else:
        Zrec = eb
    theta = dict(zip(post.names, map(float, th)))
    theta['contact_um'] = L_um - theta.get('tip_setback_um', L_um - NODE_CONTACT_UM)
    theta['zeta'] = 10 ** theta['log_zeta']
    theta['red_chi2'] = float(post.red_chi2)
    sd = np.full(Zrec.shape, np.nan)
    return dict(Zrec=Zrec * scale, Zeb=eb * scale, sd=sd, sd_gp=sd, sd_par=sd,
                ell=None, theta=theta, gain=complex(post.gain) * scale)
