"""Load the 2026-09-17 dense sweep and put it on the EB model's grid."""
import numpy as np

OM1 = 1.87510407 ** 2      # scaled frequency of the free fundamental

NPZ = '/mnt/user-data/uploads/091726 Softprobe/DenseSweep_PPPCONTAU_100nN.npz'
F_LO, F_HI = 50e3, 80e3          # band around contact mode 1 (63.47 kHz)


def load_dense(npz=NPZ, f_lo=F_LO, f_hi=F_HI):
    d = np.load(npz)
    x_um = np.asarray(d['x_um'], float)          # 75 .. 445, ascending
    f = np.asarray(d['freq'][0], float)
    Z = np.asarray(d['Z'], complex)              # (npos, nfreq)
    m = (f >= f_lo) & (f <= f_hi)
    return dict(x_um=x_um, f_Hz=f[m], Z=Z[:, m],
                L_um=float(d['probe_L_um']), load_nN=float(d['load_nN']),
                step_um=float(d['step_um']))


def to_model_grid(dense, model):
    """Interpolate the measured spectra onto the model's frequency grid (Hz).

    Returns (xi, Zg, scale) with Zg normalised so max|Zg| = 1 — the EB fit's
    A0 prior is a log10 box around unity, so the data has to arrive at that scale.
    """
    f_model = model.omega / OM1 * model.geom.f0_hz
    fm = dense['f_Hz']
    Zg = np.empty((dense['Z'].shape[0], f_model.size), complex)
    for i, z in enumerate(dense['Z']):
        Zg[i] = (np.interp(f_model, fm, z.real, left=np.nan, right=np.nan)
                 + 1j * np.interp(f_model, fm, z.imag, left=np.nan, right=np.nan))
    if not np.isfinite(Zg).all():
        raise ValueError('model frequency grid runs outside the measured band; '
                         'narrow omega_lo/omega_hi')
    scale = np.abs(Zg).max()
    return dense['x_um'] / dense['L_um'], Zg / scale, scale


def estimate_sigma(Zg, half=8):
    """Noise scale from the high-frequency part of each spectrum.

    A running median over `half` points tracks the (smooth) resonance line shape;
    what is left is measurement noise. Uses the MAD so a sharp peak does not
    inflate it.
    """
    from scipy.ndimage import uniform_filter1d
    r = Zg - (uniform_filter1d(Zg.real, 2 * half + 1, axis=1)
              + 1j * uniform_filter1d(Zg.imag, 2 * half + 1, axis=1))
    a = np.abs(r).ravel()
    return float(1.4826 * np.median(np.abs(a - np.median(a))) + np.median(a))
