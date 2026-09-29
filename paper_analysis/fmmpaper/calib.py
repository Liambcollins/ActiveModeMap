"""InvOLS drift: rebuild the calibration timeline of an autonomous campaign from its raw files.

Every laser move in the SCM-PIT-B campaigns ended with a force curve that re-measured
InvOLS. Those values sit in the force-curve notes (F*.ibw) but the reductions used
the pre-flight InvOLS(x) curve, measured once at the start. This module:

1. reads InvOLS / AmpInvOLS / date-time from every force-curve note (plain-text
   scan of the .ibw; no igor2 needed), cached to results/fc_notes_<campaign>.csv;
2. parses the stage logs into laser-position intervals (run_series positions,
   imaging stops, probe checks);
3. assigns each force curve to the position it was taken at. **A force-curve file is
   time-stamped when the NEXT force curve starts**, so a curve whose time falls in
   position k's interval belongs to position k-1. The one-position lag is
   empirical: without it, walk ratios fall from 0.9 to 0.45 along the lever; with it,
   they are flat to 0.7-2.5 %. Curves the logs report explicitly (imaging stops,
   probe checks, close-out) are matched exactly by value instead;
4. returns ratio = InvOLS(measured then, there) / InvOLS(pre-flight, same x).

The correction for any checkpoint-based pm/V number is that ratio at the stage and
position the spectra were taken.
"""
from __future__ import annotations

import datetime as dt
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from . import config

NOTE_KEYS = ("InvOLS", "AmpInvOLS", "SpringConstant", "Date", "Time", "DeflectionSetpointVolts")
_TS = re.compile(r"^(\d\d):(\d\d):(\d\d)\s")


# ----------------------------------------------------------------------------- notes
def read_note(path) -> dict:
    t = Path(path).read_bytes().decode("latin-1")
    out = {}
    for k in NOTE_KEYS:
        m = re.search(r"(?:^|\r)" + k + r":([^\r]*)\r", t)
        out[k] = m.group(1).strip() if m else ""
    return out


def fc_notes(campaign: str, refresh: bool = False) -> pd.DataFrame:
    """All force-curve notes of a campaign folder (e.g. 'DomainsB_SCMPIT_R2'), cached."""
    cache = config.RESULTS_DIR / f"fc_notes_{campaign}.csv"
    if cache.exists() and not refresh:
        df = pd.read_csv(cache)
    else:
        rows = []
        for f in sorted(glob.glob(str(config.data_root() / campaign / "*" / "F*.ibw"))):
            p = Path(f)
            rows.append(dict(campaign=campaign, stage=p.parent.name, file=p.name, **read_note(p)))
        df = pd.DataFrame(rows)
        df.to_csv(cache, index=False)
    for k in ("InvOLS", "AmpInvOLS", "SpringConstant"):
        df[k] = pd.to_numeric(df[k], errors="coerce")
    df["t"] = pd.to_datetime(df.Date + " " + df.Time, format="%Y-%m-%d %I:%M:%S %p")
    return df.sort_values("t").reset_index(drop=True)


# ----------------------------------------------------------------------------- logs
def _parse_log(path, day0, last):
    iv, ex = [], []
    cur = run_start = prev_end = None
    pos, pending_x = {}, None
    stage = Path(path).parent.name
    for line in open(path, encoding="utf-8", errors="replace"):
        m = _TS.match(line)
        if m:
            t = day0 + dt.timedelta(hours=int(m[1]), minutes=int(m[2]), seconds=int(m[3]))
            while last is not None and t < last - dt.timedelta(hours=2):
                t += dt.timedelta(days=1)
            cur = last = t
        if line.startswith("fixed design"):
            run_start = prev_end = cur
            pos = {}
        m = re.search(r"=== position (\d+): x = ([\d.]+) um", line)
        if m:
            pos[int(m[1])] = float(m[2])
        m = re.search(r"(\d+)/\d+ fixed positions done \(([\d.]+) min elapsed\)", line)
        if m and run_start is not None:
            end = run_start + dt.timedelta(minutes=float(m[2]))
            iv.append(dict(t0=prev_end, t1=end, x_um=pos.get(int(m[1])), stage=stage))
            prev_end = end
        m = re.search(r"laser -> stage x = ([\d.]+) um", line)
        if m:
            pending_x = float(m[1])
        m = re.search(r"^\d\d:\d\d:\d\d\s+InvOLS ([\d.eE+-]+) m/V", line)
        if m and pending_x is not None:
            ex.append(dict(t=cur, x_um=pending_x, invols=float(m[1]), stage=stage)); pending_x = None
        m = re.search(r"PROBE CHECK.*InvOLS ([\d.eE+-]+)", line)
        if m:
            ex.append(dict(t=cur, x_um=232.0, invols=float(m[1]), stage=stage))   # probe checks run at the free end
        m = re.search(r"final InvOLS at ([\d.]+) um: ([\d.eE+-]+)", line)
        if m:
            ex.append(dict(t=cur, x_um=float(m[1]), invols=float(m[2]), stage=stage))
    return iv, ex, last


def _first_seconds(path):
    for line in open(path, encoding="utf-8", errors="replace"):
        m = _TS.match(line)
        if m:
            return int(m[1]) * 3600 + int(m[2]) * 60 + int(m[3])
    return None


def preflight_curve(campaign: str):
    pf = json.load(open(config.data_root() / campaign / "01_preflight" / "preflight_result.json"))
    xs = np.array(sorted(float(k) for k, v in pf["invols_by_x"].items() if v not in (None, "None")))
    vs = np.array([float(pf["invols_by_x"][f"{k:.1f}"]) for k in xs])
    return xs, vs, (lambda x: np.exp(np.interp(np.asarray(x, float), xs, np.log(vs))))


def timeline(campaign: str, day0: dt.datetime, refresh: bool = False, lag: int = 1) -> pd.DataFrame:
    """InvOLS measured at every laser stop of a campaign, with ratio to the pre-flight curve.

    ``day0`` is the date and hour the campaign started (logs carry times only).
    ``lag`` = 1 applies the one-position file-timestamp lag (default); 0 shows the naive
    assignment for comparison.
    """
    xs, vs, ref = preflight_curve(campaign)
    notes = fc_notes(campaign, refresh)
    notes = notes[notes.InvOLS < 1.05 * vs.max()]           # drop rejected out-of-range retries
    logs = [p for p in glob.glob(str(config.data_root() / campaign / "*" / "*_log.txt"))
            if "01_preflight" not in p and "_relay" not in p]
    start_s = day0.hour * 3600
    order = sorted(logs, key=lambda p: (_first_seconds(p) or 0) + (86400 if (_first_seconds(p) or 0) < start_s else 0))
    IV, EX, last = [], [], None
    base = dt.datetime(day0.year, day0.month, day0.day)
    for p in order:
        iv, ex, last = _parse_log(p, base, last or day0)
        IV += iv; EX += ex
    rows, used = [], set()
    for e in EX:                                             # exact matches by value
        c = notes[np.isclose(notes.InvOLS, e["invols"], rtol=2e-4) & ~notes.file.isin(used)]
        if len(c):
            r = c.iloc[int(np.argmin(np.abs((c.t - e["t"]).dt.total_seconds())))]
            used.add(r.file)
            rows.append(dict(stage=e["stage"], kind="logged", file=r.file, t=e["t"], x_um=e["x_um"],
                             invols=float(r.InvOLS)))
    IV = sorted([v for v in IV if v["x_um"] is not None], key=lambda v: v["t0"])
    rest = notes[~notes.file.isin(used)]
    slack = dt.timedelta(seconds=5)
    for _, r in rest.iterrows():                            # interval match with one-position lag
        k = next((i for i, v in enumerate(IV) if v["t0"] - slack <= r.t <= v["t1"] + slack), None)
        if k is None or k - lag < 0:
            continue
        v = IV[k - lag]
        rows.append(dict(stage=v["stage"], kind="interval", file=r.file, t=v["t1"], x_um=v["x_um"],
                         invols=float(r.InvOLS)))
    tl = pd.DataFrame(rows)
    tl = tl.drop_duplicates(subset=["stage", "t", "x_um"], keep="last")
    tl["ref"] = ref(tl.x_um.values)
    tl["ratio"] = tl.invols / tl.ref
    tl["campaign"] = campaign
    return tl.sort_values("t").reset_index(drop=True)


def stage_factors(tl: pd.DataFrame) -> pd.DataFrame:
    """Median and spread of the InvOLS ratio per stage, and per (stage, position)."""
    g = tl.groupby("stage").ratio
    return pd.DataFrame(dict(n=g.size(), t_start=tl.groupby("stage").t.min(),
                             median=g.median(), sd=g.std(), lo=g.min(), hi=g.max())).sort_values("t_start")


def factor_at(tl: pd.DataFrame, stage: str, x_um) -> np.ndarray:
    """Correction factor for spectra taken in ``stage`` at positions ``x_um``.

    Uses the force curve measured at that stop when there is one; otherwise the
    stage median.
    """
    d = tl[tl.stage == stage]
    med = float(d.ratio.median()) if len(d) else np.nan
    out = []
    for x in np.atleast_1d(x_um):
        m = d[np.isclose(d.x_um, x, atol=0.05)]
        out.append(float(m.ratio.median()) if len(m) else med)
    return np.array(out)


# ----------------------------------------------------------------------------- frame drift
def static_shape_x0(x_um, invols, L_um: float = 226.5, grid=np.arange(-30, 60, 0.05)):
    """Clamp position in stage coordinates from one walk's InvOLS(x), lever length fixed.

    InvOLS(x) of a tip-loaded cantilever follows 1/[(x-x0)^2 (3L-(x-x0))] regardless of
    the contact stiffness, so x0 is a direct measurement of where the laser really is.
    Returns (x0, gain, rms_log_residual).
    """
    x = np.asarray(x_um, float); y = 1.0 / np.asarray(invols, float)
    best = None
    for x0 in grid:
        s = x - x0
        if s.min() <= 0 or s.max() > L_um:
            continue
        m = s ** 2 * (3 * L_um - s)
        g = np.sum(y * m) / np.sum(m * m)
        r = float(np.sqrt(np.mean(np.log(y / (g * m)) ** 2)))
        if best is None or r < best[2]:
            best = (float(x0), float(g), r)
    return best


def frame_anchors(campaign: str, tl: pd.DataFrame, L_um: float = 226.5,
                  exclude=("dense_grid", "load_ladder", "bias_vs_load")) -> pd.DataFrame:
    """x0(t) from every short multi-position walk (pre-flight, surveys, wideband, close-out).

    Dense maps are excluded: a monotonic free-end-to-base walk confounds time and
    position. Load-ladder walks are excluded because InvOLS there also depends on load.
    """
    xs, vs, _ = preflight_curve(campaign)
    x0, g, r = static_shape_x0(xs, vs, L_um)
    pf_t = tl.t.min() - pd.Timedelta(minutes=20)
    rows = [dict(stage="01_preflight", t=pf_t, x0=x0, gain=g, rms=r, n=len(xs))]
    for st, d in tl.groupby("stage"):
        if any(e in st for e in exclude):
            continue
        d = d[(d.ratio > 0.7) & (d.ratio < 1.6)]
        dd = d.groupby("x_um").agg(invols=("invols", "median"), t=("t", "median")).reset_index()
        if dd.x_um.nunique() < 5:
            continue
        x0, g, r = static_shape_x0(dd.x_um.values, dd.invols.values, L_um)
        rows.append(dict(stage=st, t=dd.t.median(), x0=x0, gain=g, rms=r, n=len(dd)))
    a = pd.DataFrame(rows).sort_values("t").reset_index(drop=True)
    a["gain_rel"] = a.gain / a.gain.iloc[0]
    return a


def x0_at(anchors: pd.DataFrame, t) -> np.ndarray:
    """Clamp position at time(s) t by linear interpolation between frame anchors."""
    ta = anchors.t.values.astype("datetime64[s]").astype(float)
    tq = np.atleast_1d(pd.to_datetime(t)).astype("datetime64[s]").astype(float)
    return np.interp(tq, ta, anchors.x0.values)


# ----------------------------------------------------------------------------- tune-note InvOLS
def tune_invols(folder, pattern="Tune_*_DCp0p000V_*_S1_*.txt", L_um: float = 451.8, x0_um=None):
    """InvOLS per position from the saved tune notes of a bias survey (e.g. the soft-probe surveys).

    The tune note carries the InvOLS in force when the spectrum was taken, i.e. that stop's own
    force curve. (The survey *log* prints the previous stop's value -- the same one-position lag as
    the force-curve files -- so do not read InvOLS from the log.) Positions come from the
    ``_X<um*1000>_`` field of the file name. Positions without a note are filled from the
    static-shape fit 1/InvOLS = g u^2 (3A - u), u = x - x0, A = tip position from the clamp.
    Returns (fn, table): fn(x) interpolates InvOLS in m/V; table lists measured and filled values.
    """
    folder = Path(folder)
    rows = []
    for f in sorted(folder.glob(pattern)):
        m = re.search(r"_X(\d{6})_", f.name)
        if m:
            b = f.read_bytes(); txt = b[-60000:].decode("latin-1")          # the note sits after the waves
            v = re.search(r"(?:\\r|\r|^)InvOLS:\s*([-+0-9.eE]+)", txt)     # deflection InvOLS (not AmpInvOLS)
            rows.append(dict(x_um=int(m[1]) / 1000, invols=float(v[1]), file=f.name))
    t = pd.DataFrame(rows).drop_duplicates("x_um").sort_values("x_um").reset_index(drop=True)
    if x0_um is None:                                   # fit the clamp position (laser frame drifts between sessions)
        m = t.x_um.values < L_um - 40                    # stay clear of the tip, where the shape law ends
        x0_um = static_shape_x0(t.x_um.values[m], t.invols.values[m], L_um, grid=np.arange(-60, 40, 0.05))[0]
    u = t.x_um.values - x0_um
    mm = u ** 2 * (3 * L_um - u)
    g = float(np.sum(mm / t.invols.values) / np.sum(mm * mm))
    t.attrs["x0_um"] = float(x0_um)
    t["shape_resid_pct"] = 100 * ((1 / t.invols.values) / (g * mm) - 1)
    t["filled"] = False

    def fn(x, _t=t, _g=g):
        x = np.atleast_1d(np.asarray(x, float))
        v = np.interp(x, _t.x_um.values, _t.invols.values)
        out = (x < _t.x_um.min() - 0.05) | (x > _t.x_um.max() + 0.05)
        uu = x[out] - x0_um
        v[out] = 1.0 / (_g * uu ** 2 * (3 * L_um - uu))
        return v if v.size > 1 else float(v[0])
    return fn, t
