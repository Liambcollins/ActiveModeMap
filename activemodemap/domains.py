"""Locate two opposite PFM domains in a scanned image and mark them as Igor spots.

This is "option B" from the notebook-05 notes: instead of the user clicking two
spots with the Force panel's pick-spot, the phase image is segmented into its two
orientations and one robust interior point is chosen per domain.

Coordinate convention (verified on the instrument, 2026-09-17)
-------------------------------------------------------------
``root:Packages:MFP3D:Force:SpotX`` / ``SpotY`` hold spot positions **in metres,
measured from the corner of the scan frame**, so the centre of a 20 um scan reads
1e-5. Index 0 is the scan origin and user spots start at index 1, which is what
``PV("ForceSpotNumber", k)`` + ``GoToSpot()`` expects.

``GoToSpot()`` reads the geometry off the OPEN image window (offset, scan angle,
LVDT sensitivities), so the image that the spots were found in must still be open
when the run starts. Nothing here can check that for you -- run
:func:`verify_spot_moves` before engaging.
"""

from __future__ import annotations

import os
import time

import numpy as np

__all__ = [
    "load_asylum_image", "list_channels", "find_domain_spots",
    "read_spots_from_igor", "write_spots_to_igor", "verify_spot_moves",
    "plot_domain_confirmation",
]


# --------------------------------------------------------------------------- #
#  image loading                                                              #
# --------------------------------------------------------------------------- #
def _note_str(note) -> str:
    if isinstance(note, bytes):
        return note.decode("utf-8", "ignore")
    return str(note)


def _note_value(note, key, default=None):
    for line in _note_str(note).splitlines():
        if line.startswith(key + ":"):
            try:
                return float(line.split(":", 1)[1])
            except ValueError:
                return line.split(":", 1)[1].strip()
    return default


def list_channels(path):
    """Channel names stored in an Asylum image .ibw, in layer order."""
    from igor2 import binarywave
    w = binarywave.load(path)["wave"]
    labels = [x.decode() if isinstance(x, bytes) else x
              for grp in w.get("labels", []) for x in grp if x]
    if labels:
        return labels
    # fall back to the note's channel list
    chans = _note_value(w["note"], "ScanChannels")
    if isinstance(chans, str):
        return [c for c in chans.replace(";", ",").split(",") if c]
    return []


def load_asylum_image(path, channel="Phase", trace="Trace"):
    """Return ``(image, meta)`` for one channel of an Asylum scan .ibw.

    `channel` is matched case-insensitively against the layer labels, preferring
    an exact ``<channel><trace>`` match (e.g. ``PhaseTrace``) and falling back to
    any label containing `channel`.

    meta carries ``scan_size_m``, ``x_offset_m``, ``y_offset_m``, ``scan_angle``,
    ``points``, ``lines`` and ``channel`` (the label actually used).
    """
    from igor2 import binarywave
    w = binarywave.load(path)["wave"]
    data = np.asarray(w["wData"])
    if data.ndim == 2:                       # single-layer file
        data = data[:, :, None]
    labels = [x.decode() if isinstance(x, bytes) else x
              for grp in w.get("labels", []) for x in grp if x]
    if len(labels) != data.shape[2]:
        labels = labels[-data.shape[2]:] if len(labels) > data.shape[2] else labels

    want = (channel + trace).lower()
    idx, used = None, None
    for i, lab in enumerate(labels):
        if lab.lower() == want:
            idx, used = i, lab
            break
    if idx is None:
        for i, lab in enumerate(labels):
            if channel.lower() in lab.lower():
                idx, used = i, lab
                break
    if idx is None:
        raise KeyError(f"channel {channel!r} not in {labels!r}")

    note = w["note"]
    meta = dict(
        channel=used,
        scan_size_m=_note_value(note, "ScanSize", np.nan),
        x_offset_m=_note_value(note, "XOffset", 0.0),
        y_offset_m=_note_value(note, "YOffset", 0.0),
        scan_angle=_note_value(note, "ScanAngle", 0.0),
        points=data.shape[0], lines=data.shape[1],
        path=path, all_channels=labels,
    )
    return data[:, :, idx], meta


# --------------------------------------------------------------------------- #
#  segmentation                                                               #
# --------------------------------------------------------------------------- #
def _otsu(v, nbins=256):
    """Otsu threshold of a 1-D array."""
    v = v[np.isfinite(v)]
    hist, edges = np.histogram(v, bins=nbins)
    centres = 0.5 * (edges[1:] + edges[:-1])
    w0 = np.cumsum(hist)
    w1 = w0[-1] - w0
    with np.errstate(invalid="ignore", divide="ignore"):
        m0 = np.cumsum(hist * centres) / w0
        m1 = (np.cumsum((hist * centres)[::-1])[::-1] - hist * centres + hist * centres)
        m1 = (np.sum(hist * centres) - np.cumsum(hist * centres)) / w1
        between = w0 * w1 * (m0 - m1) ** 2
    between[~np.isfinite(between)] = -np.inf
    return centres[int(np.argmax(between))]


def _phase_projection(phase_deg):
    """Project a wrapped phase image onto the axis separating its two modes.

    Works for any phase offset: the image is mapped to unit vectors and the
    dominant axis of their scatter is found, so a 0/180 pair and a -90/+90 pair
    are handled identically. Returns a real image whose sign splits the domains.
    """
    z = np.exp(1j * np.deg2rad(np.asarray(phase_deg, float)))
    c, s = np.real(z), np.imag(z)
    good = np.isfinite(c) & np.isfinite(s)
    cc, ss = c[good] - c[good].mean(), s[good] - s[good].mean()
    cov = np.array([[np.mean(cc * cc), np.mean(cc * ss)],
                    [np.mean(cc * ss), np.mean(ss * ss)]])
    evals, evecs = np.linalg.eigh(cov)
    v = evecs[:, int(np.argmax(evals))]
    return (c - np.nanmean(c)) * v[0] + (s - np.nanmean(s)) * v[1]


def find_domain_spots(phase_deg, scan_size_m, border_frac=0.08,
                      min_area_frac=0.03, n_spots=2, smooth_px=3,
                      min_phase_sep_deg=60.0, min_margin_um=0.5):
    """Find one deep-interior point inside each of the two domain orientations.

    Returns a list of ``n_spots`` dicts sorted with the positive-projection
    ("up") domain first, each with:

    ``row``/``col``      pixel indices
    ``x_m``/``y_m``      metres from the scan-frame corner, ready for SpotX/SpotY
    ``margin_um``        distance to the nearest domain wall *or image edge*
    ``area_frac``        fraction of the image occupied by that domain
    ``mean_phase_deg``   circular mean phase inside the domain

    ``margin_um`` is the number to look at: a spot a micron from a wall will
    drift across it during a long run.

    Two guards, because Otsu will always return a threshold and will happily cut
    a single-domain image in half along its own noise:

    * the circular separation of the two domains' mean phases must exceed
      `min_phase_sep_deg` -- real opposite domains are ~180 deg apart, noise is a
      few degrees;
    * each spot's margin must exceed `min_margin_um`.

    Either failing raises, because marking a spot on noise and running a
    multi-hour series on it is the expensive mistake here.
    """
    from scipy import ndimage

    phase = np.asarray(phase_deg, float)
    proj = _phase_projection(phase)
    thr = _otsu(proj.ravel())
    n_r, n_c = phase.shape
    px_m = float(scan_size_m) / max(n_r, n_c)

    # keep spots away from the frame edge, where drift bites first
    border = int(round(border_frac * min(n_r, n_c)))
    valid = np.zeros_like(proj, dtype=bool)
    valid[border:n_r - border or None, border:n_c - border or None] = True

    out = []
    for sign in (+1, -1):
        raw = ((proj > thr) if sign > 0 else (proj <= thr)) & np.isfinite(proj)
        area_frac = float(raw.mean())
        if area_frac < min_area_frac:
            continue
        # knock out salt-and-pepper before measuring interiors, otherwise a few
        # percent of flipped pixels pit the domain and collapse the margin
        mask = raw
        if smooth_px and smooth_px > 1:
            mask = ndimage.median_filter(raw.astype(np.uint8), size=int(smooth_px)) > 0
            if not mask.any():
                mask = raw
        # largest connected component only -- speckle is not a domain
        lab, n = ndimage.label(mask)
        if n > 1:
            sizes = ndimage.sum(mask, lab, range(1, n + 1))
            mask = lab == (int(np.argmax(sizes)) + 1)
        # Distance to the nearest wall OR image edge: pad with False so a domain
        # running off the frame does not report a margin it does not have.
        padded = np.zeros((n_r + 2, n_c + 2), dtype=bool)
        padded[1:-1, 1:-1] = mask
        dist = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
        dist_v = np.where(valid, dist, 0.0)
        if dist_v.max() <= 0:
            dist_v = dist
        r, c = np.unravel_index(int(np.argmax(dist_v)), dist_v.shape)
        zbar = np.mean(np.exp(1j * np.deg2rad(phase[raw])))
        out.append(dict(
            row=int(r), col=int(c),
            # SpotX runs along the fast (column) axis, SpotY along the slow one
            x_m=float((c + 0.5) * px_m), y_m=float((r + 0.5) * px_m),
            margin_um=float(dist_v[r, c] * px_m * 1e6),
            area_frac=area_frac,
            mean_phase_deg=float(np.rad2deg(np.angle(zbar))),
            projection_sign=int(sign),
        ))

    if len(out) < n_spots:
        raise RuntimeError(
            f"found only {len(out)} domain(s) covering >= {min_area_frac:.0%} of the "
            "image. Either the field of view is single-domain or the phase contrast "
            "is too weak to segment -- look at the image before trusting this.")

    sep = abs((out[0]["mean_phase_deg"] - out[1]["mean_phase_deg"] + 180) % 360 - 180)
    if sep < min_phase_sep_deg:
        raise RuntimeError(
            f"the two segments' mean phases differ by only {sep:.1f} deg "
            f"(< {min_phase_sep_deg:.0f}). This image is almost certainly SINGLE "
            "domain and the split is just noise -- move to a field of view with "
            "real contrast rather than marking spots on this one.")
    thin = [i + 1 for i, s in enumerate(out[:n_spots]) if s["margin_um"] < min_margin_um]
    if thin:
        raise RuntimeError(
            f"spot(s) {thin} sit within {min_margin_um} um of a wall or the image "
            "edge, so scanner drift will carry the tip out of the domain during the "
            "run. Use a field of view with larger domains, or mark the spots by hand.")
    return out[:n_spots]


# --------------------------------------------------------------------------- #
#  Igor spot marking                                                          #
# --------------------------------------------------------------------------- #
_SPOT_DF = r"root:Packages:MFP3D:Force"


def _wave_npnts(wave):
    try:
        return int(wave.GetDimensions()[1][0])
    except Exception:
        return int(wave.GetDimensions()[1])


def read_spots_from_igor(igor):
    """Current (SpotX, SpotY) contents in metres, index 0 first."""
    df = igor.DataFolder(_SPOT_DF)
    sx, sy = df.Wave("SpotX"), df.Wave("SpotY")
    n = min(_wave_npnts(sx), _wave_npnts(sy))
    return [(sx.GetNumericWavePointValue(i), sy.GetNumericWavePointValue(i))
            for i in range(n)]


def write_spots_to_igor(igor, spots, start_index=1, verbose=True):
    """Write found spots into SpotX/SpotY starting at `start_index`.

    `spots` is the list from :func:`find_domain_spots` (or any sequence of
    ``(x_m, y_m)`` pairs). Index 0 is left alone -- it is the scan origin.
    Returns the spot numbers written, for ``PV("ForceSpotNumber", k)``.
    """
    pairs = [(s["x_m"], s["y_m"]) if isinstance(s, dict) else tuple(s) for s in spots]
    n_needed = start_index + len(pairs)
    igor.Execute(f"SetDataFolder {_SPOT_DF}")
    igor.Execute(f"Redimension/N={n_needed} SpotX, SpotY")
    numbers = []
    for k, (x_m, y_m) in enumerate(pairs, start=start_index):
        igor.Execute(f"SpotX[{k}] = {x_m:.10e}")
        igor.Execute(f"SpotY[{k}] = {y_m:.10e}")
        numbers.append(k)
        if verbose:
            print(f"  spot {k}: ({x_m * 1e6:.3f}, {y_m * 1e6:.3f}) um from the scan corner")
    igor.Execute("SetDataFolder root:")
    return numbers


def verify_spot_moves(automation, spot_numbers, expected_sep_m=None,
                      settle_s=1.0, verbose=True):
    """Hop between the marked spots with the tip withdrawn and read XY sensors.

    Catches the two failures that cost a whole run: spots that do not move the
    scanner at all (marked at the same place, or GoToSpot refusing because the
    image window is closed), and a coordinate convention that is off.

    If `expected_sep_m` is given -- the distance between the two spots in metres,
    which :func:`find_domain_spots` output supplies -- the measured sensor
    separation is converted through XLVDTSens/YLVDTSens and compared, and the
    ratio is printed. A ratio far from 1 means the convention is wrong; STOP.

    Returns ``(ok, readings)``.
    """
    gmv = automation.get_gmv()
    xs, ys = gmv["XLVDTSens"], gmv["YLVDTSens"]
    automation.withdraw()
    time.sleep(settle_s)

    order = list(spot_numbers) + [spot_numbers[0]]
    readings = []
    for s in order:
        settled = automation.goto_spot(s)
        xy = automation.read_xy_sensor()
        readings.append(dict(spot=s, settled=bool(settled), xy_V=tuple(xy)))
        if verbose:
            print(f"  spot {s}: settled={settled}  XY sensor = "
                  f"({xy[0]:+.4f}, {xy[1]:+.4f}) V")
        time.sleep(settle_s)

    ok = all(r["settled"] for r in readings)
    a, b = readings[0]["xy_V"], readings[1]["xy_V"]
    sep_m = float(np.hypot((b[0] - a[0]) * xs, (b[1] - a[1]) * ys))
    if verbose:
        print(f"  measured spot separation: {sep_m * 1e6:.3f} um")
    if sep_m < 1e-7:
        print("  ** the two spots read the same position -- they are not distinct, "
              "or GoToSpot did nothing (is the image window still open?)")
        ok = False
    if expected_sep_m:
        ratio = sep_m / float(expected_sep_m)
        if verbose:
            print(f"  expected {float(expected_sep_m) * 1e6:.3f} um  ->  ratio {ratio:.3f}")
        if not (0.8 < ratio < 1.25):
            print("  ** measured/expected separation is off by more than 25%. The spot "
                  "coordinate convention is probably wrong -- do NOT engage; mark the "
                  "spots by hand with the Force-panel pick-spot instead.")
            ok = False
    # return the scanner to the first spot
    automation.goto_spot(spot_numbers[0])
    return ok, readings


# --------------------------------------------------------------------------- #
#  confirmation figure                                                        #
# --------------------------------------------------------------------------- #
def plot_domain_confirmation(phase_deg, spots, scan_size_m, save_path=None,
                             title="PFM phase — proposed domain spots"):
    """Phase image with the proposed spots and their wall margins drawn on it."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    phase = np.asarray(phase_deg, float)
    ext_um = float(scan_size_m) * 1e6
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.6))

    im = ax[0].imshow(phase, origin="lower", extent=[0, ext_um, 0, ext_um],
                      cmap="twilight")
    fig.colorbar(im, ax=ax[0], label="phase (deg)")
    ax[0].set_title(title, fontsize=10)

    proj = _phase_projection(phase)
    ax[1].imshow(proj > _otsu(proj.ravel()), origin="lower",
                 extent=[0, ext_um, 0, ext_um], cmap="gray")
    ax[1].set_title("segmentation (white = 'up')", fontsize=10)

    for a in ax:
        for k, s in enumerate(spots, start=1):
            x, y = s["x_m"] * 1e6, s["y_m"] * 1e6
            a.add_patch(Circle((x, y), s["margin_um"], fill=False,
                               color="#e34948", lw=1.0, alpha=0.9))
            a.plot(x, y, "+", color="#e34948", ms=11, mew=1.8)
            a.text(x, y + s["margin_um"] + 0.4, f"spot {k}", color="#e34948",
                   ha="center", fontsize=9)
        a.set_xlabel("x (µm)")
        a.set_ylabel("y (µm)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180)
    return fig
