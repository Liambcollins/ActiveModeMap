"""Readers and per-domain statistics for BEPyAE band-excitation frames (.hf5) and Asylum QS images (QImg*.ibw).

BE: 'BE Quick Fitting' channels = [A, f_res, Q, phase(rad), unused]. A is the SHO amplitude (raw V of the AC
channel), which BEPyAE reports as peak/Q (checked by `check_A_is_peak_over_Q`). Bin frequencies are
R*_acquisition/BE_pulse_parm_indicator_06[indicator_02] ('BE Parameters/frequency' is NOT the analysed band).
QS: AmplitudeRetrace is already in metres (AmpInvOLS = InvOLS/32 at the time of the scan).
"""
from __future__ import annotations
import numpy as np
import h5py
from igor2 import binarywave as bw


def _acq_group(f):
    return [k for k in f.keys() if k.endswith("_acquisition")][0]


def load_be(path, spectra=False, sub=None):
    with h5py.File(path, "r") as f:
        q = f["BE Quick Fitting/Quick Fitting/Quick Fitting"][()]
        g = f[_acq_group(f)]; attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in g.attrs.items()}
        ind06 = g["BE_pulse_parm_indicator_06"][()]; ind02 = g["BE_pulse_parm_indicator_02"][()].astype(int)
        fb = ind06[ind02]
        out = dict(A=q[..., 0], f=q[..., 1], Q=q[..., 2], phi=q[..., 3], f_bins=fb, attrs=attrs,
                   scan_size=f["BE Parameters/scan size"][()], pulse=f["BE Parameters/pulse parameters"][()])
        out["pixel_time_s"] = float(attrs.get("pixel_time_s", out["pulse"][5]))
        if spectra:
            S = f["BE Complex Spectra/Complex Spectra/Complex Spectra"]
            out["S"] = S[sub] if sub is not None else S[()]
    return out


def note_kv(w):
    note = w["note"].decode("latin1", errors="ignore")
    return dict((l.split(":", 1)[0].strip(), l.split(":", 1)[1].strip()) for l in note.replace("\r", "\n").split("\n") if ":" in l)


def load_qs(path):
    w = bw.load(path)["wave"]; D = w["wData"]; lab = [x.decode() for x in w["labels"][2] if x]; kv = note_kv(w)
    get = lambda n: D[..., [i for i, l in enumerate(lab) if l.startswith(n)][0]].T   # Igor stores (x, y); transpose to (y, x)
    return dict(amp_m=get("Amplitude"), phase_deg=get("Phase"), height=get("Height"), note=kv,
                invols=float(kv["InvOLS"]), drive_V=float(kv["DriveAmplitude"]), f_drive=float(kv["DriveFrequency"]),
                scan_size=float(kv["ScanSize"]), setpoint_V=float(kv.get("SetpointVolts", kv.get("DeflectionSetpointVolts", "nan"))))


def split_domains(phi_rad, n_iter=30):
    """Two-cluster split on the phase circle. Returns labels (0/1; 0 = the larger domain) and cluster centres."""
    z = np.exp(1j * np.asarray(phi_rad)); c = np.array([np.exp(1j * np.nanpercentile(phi_rad, 10)), np.exp(1j * np.nanpercentile(phi_rad, 90))])
    for _ in range(n_iter):
        lab = np.argmin(np.abs(z[..., None] - c), -1)
        c = np.array([np.nanmean(z[lab == k]) if np.any(lab == k) else c[k] for k in (0, 1)]); c = c / np.abs(c)
    if (lab == 1).sum() > (lab == 0).sum(): lab = 1 - lab; c = c[::-1]
    return lab, np.degrees(np.angle(c))


def domain_stats(amp, phi_rad, lab, extra=None, trim=2):
    """Per-domain medians of amplitude, complex half-difference contrast |<z>_0 - <z>_1|/2, fraction, flip angle.
    Pixels within `trim` px of a wall are excluded from the medians."""
    from scipy.ndimage import binary_erosion
    out = {}; z = amp * np.exp(1j * phi_rad)
    for k in (0, 1):
        m = binary_erosion(lab == k, iterations=trim) if trim else (lab == k)
        if m.sum() < 50: m = lab == k
        if m.sum() == 0: out[k] = dict(n=0, frac=0.0, amp_med=np.nan, z_med=np.nan, **{nm + "_med": np.nan for nm in (extra or {})}); continue
        d = dict(n=int(m.sum()), frac=float((lab == k).mean()), amp_med=float(np.nanmedian(amp[m])),
                 z_med=complex(np.nanmedian(z[m].real) + 1j * np.nanmedian(z[m].imag)))
        for name, arr in (extra or {}).items(): d[name + "_med"] = float(np.nanmedian(arr[m]))
        out[k] = d
    if out[1]["n"] >= 50:
        out["contrast"] = float(abs(out[0]["z_med"] - out[1]["z_med"]) / 2)
        out["flip_deg"] = float(np.degrees(np.angle(out[0]["z_med"] * np.conj(out[1]["z_med"]))) % 360)
    return out


def check_A_is_peak_over_Q(path, n=400, seed=0):
    """max|S| over the band vs A*Q on random pixels. Returns median and IQR of the ratio."""
    r = np.random.default_rng(seed)
    with h5py.File(path, "r") as f:
        q = f["BE Quick Fitting/Quick Fitting/Quick Fitting"]; S = f["BE Complex Spectra/Complex Spectra/Complex Spectra"]
        ny, nx = q.shape[:2]; ii = r.integers(0, ny, n); jj = r.integers(0, nx, n); rat = []
        for i, j in zip(ii, jj):
            A, Q = q[i, j, 0], q[i, j, 2]; s = np.abs(S[i, j]); rat.append(s.max() / (A * Q))
    rat = np.array(rat); return dict(median=float(np.median(rat)), p25=float(np.percentile(rat, 25)), p75=float(np.percentile(rat, 75)))
