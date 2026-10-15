"""Quantitative single-frequency PFM frames and their conversion to d_eff.

Igor control (frame parameters, drive, one down-scan) follows the R3 campaign
notebook; conversion follows the paper: a frame's AmplitudeRetrace is already in
metres (InvOLS applied on the instrument at that laser stop), so

    d_eff (pm/V) = A / V_ac / E          with E = 1 for a quasi-static frame,
                                          E = E_map(x_img) for a CR1 frame.

Domain medians are reported separately; their mean is the quoted value, their
difference a check on the domain-potential term.
"""
from __future__ import annotations

import glob
import os
import time

import numpy as np

FRAME_KEYS = ("ScanSize", "XOffset", "YOffset", "ScanAngle", "ScanPoints", "ScanLines", "ScanRate")


# ----------------------------------------------------------------------------------- #
#  Igor side                                                                            #
# ----------------------------------------------------------------------------------- #
def pv(inst, key, value):
    inst.a.igor.Execute(f'PV("{key}", {value})')


def verify(inst, expected, tol=0.02, abstol=None, say=print):
    """Read MasterVariables back and compare; returns the keys that did not take."""
    g, bad = inst.a.get_gmv(), []
    for k, want in expected.items():
        got = g.get(k)
        if got is None:
            say(f"    {k:<14} not in MasterVariables"); continue
        lim = abstol if abstol is not None else tol * max(abs(want), 1e-12)
        ok = abs(got - want) <= lim
        say(f"    {k:<14} asked {want:<12.6g} reads {got:<12.6g} {'ok' if ok else '** MISMATCH'}")
        if not ok:
            bad.append(k)
    return bad


def read_frame(inst):
    """The scan frame currently set in the MasterPanel."""
    g = inst.a.get_gmv()
    return {k: g.get(k) for k in FRAME_KEYS}


def set_frame(inst, frame, say=print):
    for k in FRAME_KEYS:
        if frame.get(k) is not None:
            pv(inst, k, frame[k])
    time.sleep(1.5)
    bad = verify(inst, {k: frame[k] for k in ("ScanSize", "ScanPoints", "ScanLines", "ScanRate", "ScanAngle")
                        if frame.get(k) is not None}, say=say)
    bad += verify(inst, {k: frame[k] for k in ("XOffset", "YOffset") if frame.get(k) is not None}, abstol=5e-8, say=say)
    if [k for k in bad if k in ("ScanSize", "XOffset", "YOffset", "ScanAngle")]:
        raise RuntimeError(f"scan frame did not take: {bad}")


def is_scanning(inst):
    return bool(inst.a.igor.DataFolder(r"root:packages:MFP3D").Wave("OutWaves").GetTextWavePointValue(0, 0))


def grab_frame(inst, base, timeout_s=1200, say=print):
    """Start one down-scan with Igor base name `base`, wait for it, return the new .ibw path."""
    igor = inst.a.igor
    inst.a.set_folder()
    igor.Execute(f'root:Packages:MFP3D:Main:Variables:BaseName = "{base}"'); igor.Execute("ARCheckSuffix()")
    pat = os.path.join(inst.a.file_loc, f"{base}*.ibw")
    before = {p: os.path.getmtime(p) for p in glob.glob(pat)}
    t0 = time.time(); inst.a.ex("DownScan_0", "MasterPanel"); time.sleep(10)
    while is_scanning(inst) and time.time() - t0 < timeout_s:
        time.sleep(2)
    if time.time() - t0 >= timeout_s:
        say("    WARNING: scan timeout")
    time.sleep(3)
    new = [p for p in glob.glob(pat) if p not in before or os.path.getmtime(p) > before[p]]
    if not new:
        raise RuntimeError(f"no new {base}*.ibw written")
    path = max(new, key=os.path.getmtime); last = -1
    for _ in range(20):
        s = os.path.getsize(path)
        if s == last and s > 0:
            break
        last = s; time.sleep(0.5)
    say(f"    -> {os.path.basename(path)} ({os.path.getsize(path) / 1e6:.2f} MB, {(time.time() - t0) / 60:.1f} min)")
    return path


def single_frequency_frame(inst, base, v_dc, f_drive_Hz, v_ac, timeout_s=1200, say=print):
    """One frame at a fixed drive frequency and DC bias, drive verified before the scan."""
    say(f"--- frame {base}: V_dc {v_dc:+.3f} V, drive {f_drive_Hz / 1e3:.3f} kHz, {v_ac} V ---")
    inst.set_dc_bias(v_dc); pv(inst, "DriveFrequency", f_drive_Hz); pv(inst, "DriveAmplitude", v_ac); time.sleep(1)
    if verify(inst, {"DriveFrequency": f_drive_Hz, "DriveAmplitude": v_ac}, tol=0.002, say=say):
        raise RuntimeError("drive did not take")
    return grab_frame(inst, base, timeout_s, say)


# ----------------------------------------------------------------------------------- #
#  conversion                                                                           #
# ----------------------------------------------------------------------------------- #
def read_ibw_channels(path):
    """Asylum image .ibw -> dict(name -> 2-D array) using igor2."""
    from igor2.binarywave import load as ibw_load
    w = ibw_load(str(path))["wave"]
    data = np.asarray(w["wData"], float)
    note = w.get("note", b"")
    note = note.decode("latin-1", "ignore") if isinstance(note, bytes) else str(note)
    labels = [l.decode("latin-1", "ignore") if isinstance(l, bytes) else str(l)
              for l in (w.get("labels") or [[], [], [], []])[2]] if w.get("labels") else []
    labels = [l for l in labels if l]
    out = {}
    if data.ndim == 3:
        for k in range(data.shape[2]):
            name = labels[k] if k < len(labels) else f"ch{k}"
            out[name] = data[:, :, k]
    else:
        out["ch0"] = data
    return out, note


def frame_deff(path, drive_V, E=1.0, amp_channel=None, phase_channel=None):
    """d_eff per pixel (pm/V) from an amplitude channel in metres, plus domain medians.

    Domains are split on the phase channel (two clusters of the phase unit vector); the
    pixels nearest each cluster centre form the domain masks. Returns a dict with the
    per-pixel map, the two domain medians, their mean and the contrast.
    """
    ch, note = read_ibw_channels(path)
    names = list(ch)
    amp = ch[amp_channel] if amp_channel else next((ch[n] for n in names if "Amplitude" in n), None)
    ph = ch[phase_channel] if phase_channel else next((ch[n] for n in names if "Phase" in n), None)
    if amp is None:
        raise ValueError(f"no amplitude channel in {os.path.basename(path)}; channels: {names}")
    d = amp / drive_V / E * 1e12
    out = dict(path=path, channels=names, deff_map=d, median_all=float(np.nanmedian(d)))
    if ph is not None:
        v = np.exp(1j * np.radians(ph))
        # two-cluster split on the phase circle (k-means on the unit vector, 10 iterations)
        c1, c2 = v.ravel()[0], v.ravel()[np.argmax(np.abs(v.ravel() - v.ravel()[0]))]
        for _ in range(10):
            m1 = np.abs(v - c1) <= np.abs(v - c2)
            if m1.all() or (~m1).all():
                break
            c1, c2 = v[m1].mean(), v[~m1].mean()
        m1 = np.abs(v - c1) <= np.abs(v - c2)
        cols = np.arange(d.shape[1])[None, :] * np.ones(d.shape)
        if cols[m1].mean() > cols[~m1].mean():       # domain 1 = the one on the left, by geometry
            m1 = ~m1
        d1, d2 = float(np.nanmedian(d[m1])), float(np.nanmedian(d[~m1]))
        out.update(domain1_median=d1, domain2_median=d2, mean_of_domains=0.5 * (d1 + d2),
                   domain_ratio=d1 / d2 if d2 else np.nan, frac_domain1=float(m1.mean()),
                   phase_sep_deg=float(abs(np.degrees(np.angle(c1 * np.conj(c2))))))
    return out


# ----------------------------------------------------------------------------------- #
#  band excitation: acquisition adapter and conversion                                 #
# ----------------------------------------------------------------------------------- #
def load_be_acquire(spec):
    """Resolve "module:function" (cfg.imaging.be_acquire) to a callable, or raise with a hint."""
    import importlib
    mod, _, fn = spec.partition(":")
    try:
        m = importlib.import_module(mod)
    except ImportError as e:
        raise RuntimeError(f"BE adapter module {mod!r} not importable ({e}). Copy activemodemap/smm/be_adapter_template.py "
                           f"to {mod}.py next to the notebook and fill in be_frame() with your BE call.") from e
    f = getattr(m, fn or "be_frame", None)
    if f is None:
        raise RuntimeError(f"{mod} has no function {fn or 'be_frame'}")
    return f


def _sho(f, f0, Q):
    """Unit-amplitude SHO, H(f0) = 1 (the convention of the hafnia analysis)."""
    return (1j * f * f0 / Q) / (f0 ** 2 - f ** 2 + 1j * f * f0 / Q)


def sho_fit(f, H, f0_guess=None, Q_grid=None):
    """Fit a complex spectrum to a·SHO(f; f0, Q) without scipy: grid over (f0, Q), linear in a.

    Returns dict(a [complex amplitude AT f0], f0_Hz, Q, rel_resid). Coarse grid, then a local refinement.
    """
    f = np.asarray(f, float); H = np.asarray(H, complex)
    Qs = np.asarray(Q_grid if Q_grid is not None else np.logspace(1, 3.5, 60))
    f0s = f if f0_guess is None else f[np.abs(f - f0_guess) <= 0.1 * f0_guess]
    best = (np.inf, None)
    for f0 in f0s[:: max(1, f0s.size // 60)]:
        for Q in Qs:
            L = _sho(f, f0, Q); a = np.vdot(L, H) / np.vdot(L, L); r = np.sum(np.abs(H - a * L) ** 2)
            if r < best[0]:
                best = (r, (f0, Q, a))
    f0, Q, a = best[1]
    for _ in range(3):                     # refine on finer local grids
        df = (f[1] - f[0]); f0s = f0 + np.linspace(-2 * df, 2 * df, 21); Qs2 = Q * np.logspace(-0.15, 0.15, 21)
        for ff in f0s:
            for QQ in Qs2:
                L = _sho(f, ff, QQ); aa = np.vdot(L, H) / np.vdot(L, L); r = np.sum(np.abs(H - aa * L) ** 2)
                if r < best[0]:
                    best = (r, (ff, QQ, aa))
        f0, Q, a = best[1]
    return dict(a=complex(a), f0_Hz=float(f0), Q=float(Q), rel_resid=float(np.sqrt(best[0] / np.sum(np.abs(H) ** 2))))


def read_be_frame(path):
    """BE frame file -> (freq_Hz [nf], Z [ny, nx, nf] complex). Accepts an .npz with 'freq_Hz'/'Z' (or
    'spectra'), or an .hf5 with a 3-D complex dataset (or real+imag pair) and 'f_center_Hz'/'f_width_Hz'
    attributes or a 1-D frequency dataset. Raise with the layout if it cannot be read."""
    path = str(path)
    if path.lower().endswith(".npz"):
        d = np.load(path)
        Z = d["Z"] if "Z" in d.files else d["spectra"]
        return np.asarray(d["freq_Hz"], float), np.asarray(Z, complex)
    import h5py
    with h5py.File(path, "r") as h:
        cands, freqs, attrs = [], [], dict(h.attrs)
        def visit(name, obj):
            if isinstance(obj, h5py.Dataset):
                attrs.update({f"{name}:{k}": v for k, v in obj.attrs.items()})
                if obj.ndim == 3 and obj.size > 1000:
                    cands.append(name)
                elif obj.ndim == 1 and 16 <= obj.size <= 1e5 and "freq" in name.lower():
                    freqs.append(name)
        h.visititems(visit)
        if not cands:
            raise ValueError(f"no 3-D spectral dataset in {os.path.basename(path)}")
        ds = h[cands[0]]
        Z = np.asarray(ds[()])
        if not np.iscomplexobj(Z):
            re_name = cands[0]; im_name = next((c for c in cands if c != re_name and ("imag" in c.lower() or "im" in c.lower())), None)
            Z = Z.astype(complex) if im_name is None else Z + 1j * np.asarray(h[im_name][()])
        nf = Z.shape[-1]
        fc = next((v for k, v in attrs.items() if k.split(":")[-1] == "f_center_Hz"), None)
        fw = next((v for k, v in attrs.items() if k.split(":")[-1] == "f_width_Hz"), None)
        if fc is not None and fw is not None:
            f = np.linspace(float(fc) - float(fw) / 2, float(fc) + float(fw) / 2, nf)
        elif freqs and h[freqs[0]].size == nf:
            f = np.asarray(h[freqs[0]][()], float)
        else:
            raise ValueError(f"no frequency axis for {cands[0]} ({nf} bins) in {os.path.basename(path)}")
    return f, Z


def be_deff(path, E, invols_m_per_V, drive_V, amp_divisor=32.0, band_Hz=None):
    """d_eff from a BE frame by phase-coherent domain averaging and an SHO fit.

    Per pixel the raw complex spectrum is in lock-in volts; domains are split on the phase at the
    frame-mean peak bin; the two domain-mean spectra are SHO-fitted; P = (H1(f0) - H2(f0)) / 2 and
    d_eff = |P| * InvOLS / G / V_ac / E   (SHO amplitude AT f0, so it is divided by E, not by M).
    Also returns the common mode |c| = |(H1 + H2) / 2| as the electrostatic check.
    """
    f, Z = read_be_frame(path)
    if band_Hz is not None:
        m = (f >= band_Hz[0]) & (f <= band_Hz[1]); f, Z = f[m], Z[..., m]
    mean_spec = Z.reshape(-1, f.size).mean(0)
    k = int(np.argmax(np.abs(mean_spec)))
    v = Z[..., k] / np.maximum(np.abs(Z[..., k]), 1e-30)
    c1, c2 = v.ravel()[0], v.ravel()[np.argmax(np.abs(v.ravel() - v.ravel()[0]))]
    for _ in range(10):
        m1 = np.abs(v - c1) <= np.abs(v - c2)
        if m1.all() or (~m1).all():
            break
        c1, c2 = v[m1].mean(), v[~m1].mean()
    m1 = np.abs(v - c1) <= np.abs(v - c2)
    cols = np.arange(Z.shape[1])[None, :] * np.ones(Z.shape[:2])
    if cols[m1].mean() > cols[~m1].mean():
        m1 = ~m1
    H1, H2 = Z[m1].mean(0), Z[~m1].mean(0)
    s1, s2 = sho_fit(f, H1, f[k]), sho_fit(f, H2, f[k])
    P = (s1["a"] - s2["a"]) / 2; c = (s1["a"] + s2["a"]) / 2
    conv = invols_m_per_V / amp_divisor / drive_V * 1e12 / E
    return dict(path=path, n_bins=int(f.size), f_lo_Hz=float(f[0]), f_hi_Hz=float(f[-1]),
                f0_domain1_Hz=s1["f0_Hz"], f0_domain2_Hz=s2["f0_Hz"], Q_domain1=s1["Q"], Q_domain2=s2["Q"],
                resid_domain1=s1["rel_resid"], resid_domain2=s2["rel_resid"],
                frac_domain1=float(m1.mean()), phase_sep_deg=float(abs(np.degrees(np.angle(s1["a"] * np.conj(s2["a"]))))),
                P_raw_V=abs(P), common_mode_V=abs(c), common_over_P=abs(c) / max(abs(P), 1e-30),
                deff_pm_per_V=abs(P) * conv, domain1_pm_per_V=abs(s1["a"]) * conv, domain2_pm_per_V=abs(s2["a"]) * conv,
                E_used=E, invols=invols_m_per_V)
