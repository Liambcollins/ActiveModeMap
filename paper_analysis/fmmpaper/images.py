"""PFM image frames (Asylum .ibw): read, segment domains, correct for domain contact potential.

    from fmmpaper import images
    z = images.read(path, drive_V)            # complex, pm per V_ac (AmplitudeRetrace is in metres)
    m1, m2 = images.domain_masks(z_ref)       # 2-means on the phase unit vector
    p = images.cpd_correct(z, m1, m2, V, V1, V2, r)   # per-pixel piezo response

Domain labels come from geometry (cluster with the smaller mean column index = domain 1),
not from brightness or phase sign, which can invert between tip states (R1 stage 7).
"""
from __future__ import annotations

import numpy as np

AMP_CH, PHASE_CH = 1, 3


def read(path, drive_V=1.0, channels=(AMP_CH, PHASE_CH)):
    """Complex response Z = A exp(i phi) in pm per V of drive. AmplitudeRetrace is already in
    metres (InvOLS applied on the instrument at that stop), so no InvOLS/32 factor here."""
    from igor2.binarywave import load as ibw_load
    d = np.asarray(ibw_load(str(path))["wave"]["wData"], float)
    return d[:, :, channels[0]] * np.exp(1j * np.radians(d[:, :, channels[1]])) / drive_V * 1e12


def domain_masks(z_ref, seed=0):
    from sklearn.cluster import KMeans
    ph = np.angle(z_ref)
    v = np.column_stack([np.cos(ph.ravel()), np.sin(ph.ravel())])
    lab = KMeans(n_clusters=2, n_init=10, random_state=seed).fit_predict(v).reshape(ph.shape)
    m0 = lab == 0
    cols = np.arange(ph.shape[1])[None, :] * np.ones(ph.shape)
    return (m0, ~m0) if cols[m0].mean() < cols[~m0].mean() else (~m0, m0)


def interior(mask, erode=4):
    from scipy.ndimage import binary_erosion
    e = binary_erosion(mask, iterations=erode)
    return e if e.sum() >= 50 else mask


def domain_contrast(z, m1, m2):
    """Two-domain piezoresponse of a frame, P = (<z>_1 - <z>_2) / 2 (complex)."""
    return (z[m1].mean() - z[m2].mean()) / 2


def cpd_correct(z, m1, m2, V, V1, V2, r_bP, delta):
    """Per-pixel piezo response with the domain contact potentials removed.

    Model (per domain s = +1, -1):  z = s*p + b (V - V_s).  The electrostatic slope b is taken
    from the survey ratio r = b/P at this lever position (complex; rotates b into the image's
    phase frame through the image's own P). With P_img = p + delta*b and b = r*P_survey-like,
    b = r * P_img / (1 + delta * r).
    Returns p per pixel (complex); |p| is the corrected piezoresponse (pm/V for QS frames).
    """
    P = domain_contrast(z, m1, m2)
    b = r_bP * P / (1 + delta * r_bP)
    return np.where(m1, z - b * (V - V1), -(z - b * (V - V2)))
