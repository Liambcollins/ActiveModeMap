"""Loaders. Every array leaves here in one convention:

* ``Z`` is complex, in raw lock-in volts (checkpoints) -- use ``units`` to convert.
* Phase is in the **model convention** (phase decreases through a resonance).
  Files written before the 2026-09-17 phase fix are Igor-raw and are conjugated
  on load; the conversion is recorded in ``meta['phase_converted']``.
* Positions are stage micrometres as logged (``x_um``). Distance from the clamp is
  ``x_um - probe.clamp_offset_um``; the SCM-PIT-A dense map additionally carries the
  InvOLS-ruler corrected axis ``x_true_um`` (see ``axis.py``).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .registry import DATASETS, PROBES, Dataset, Probe


# --------------------------------------------------------------------------- containers
@dataclass
class Series:
    """Complex spectra Z[condition, position, frequency] plus their conditions."""
    key: str
    probe: Probe
    x_um: np.ndarray
    freq_Hz: np.ndarray
    Z: np.ndarray
    conditions: pd.DataFrame
    meta: dict = field(default_factory=dict)

    # ---- convenience -------------------------------------------------------
    @property
    def x_clamp_um(self) -> np.ndarray:
        return self.x_um - self.probe.clamp_offset_um

    def band(self, name_or_range) -> np.ndarray:
        """Boolean frequency mask for a named band ('CR1', 'QS', ...) or (lo, hi) in Hz."""
        if isinstance(name_or_range, str):
            rng = self.probe.qs_band_Hz if name_or_range.upper() == "QS" \
                else self.probe.bands_Hz[name_or_range.upper()]
        else:
            rng = name_or_range
        return (self.freq_Hz >= rng[0]) & (self.freq_Hz <= rng[1])

    def where(self, **match) -> np.ndarray:
        """Indices of conditions matching all given column values, e.g. where(spot=1)."""
        m = np.ones(len(self.conditions), bool)
        for k, v in match.items():
            m &= np.isclose(self.conditions[k].astype(float), float(v))
        return np.flatnonzero(m)

    def map(self, cond: int = 0) -> np.ndarray:
        """Z[position, freq] for one condition (dense maps have a single condition)."""
        return self.Z[cond]

    def __repr__(self) -> str:
        c = ", ".join(f"{k}={sorted(self.conditions[k].dropna().unique().tolist())}"
                      for k in self.conditions.columns)
        return (f"Series({self.key!r}, probe={self.probe.key}, Z{self.Z.shape}, "
                f"x {self.x_um.min():.1f}-{self.x_um.max():.1f} um, "
                f"f {self.freq_Hz.min()/1e3:.1f}-{self.freq_Hz.max()/1e3:.1f} kHz, {c})")


# --------------------------------------------------------------------------- helpers
def path_of(key: str) -> Path:
    return config.data_root() / DATASETS[key].path


def _to_model_phase(Z: np.ndarray, convention: str, want: str):
    if want == "raw" or convention == want:
        return Z, False
    if convention in ("igor-raw", "unknown") and want == "model":
        return np.conj(Z), True
    raise ValueError(f"cannot convert phase convention {convention!r} -> {want!r}")


def phase_slope_through_peak(freq_Hz, z, band) -> float:
    """Phase change (deg) across the -3 dB width of the largest peak in ``band``.

    Negative = model convention (phase falls through resonance); positive = Igor-raw.
    Use this to check a file's convention before trusting a phase-sensitive result.
    """
    m = (freq_Hz >= band[0]) & (freq_Hz <= band[1])
    a = np.abs(z[m]); ph = np.unwrap(np.angle(z[m])); k = int(np.argmax(a))
    half = a[k] / np.sqrt(2); lo = k; hi = k
    while lo > 0 and a[lo] > half:
        lo -= 1
    while hi < a.size - 1 and a[hi] > half:
        hi += 1
    return float(np.degrees(ph[hi] - ph[lo]))


# --------------------------------------------------------------------------- loaders
def load_series_file(path, probe: Probe, key: str = "", phase: str = "model") -> Series:
    d = np.load(path, allow_pickle=True)
    Z = np.asarray(d["Z"])
    if Z.ndim == 2:
        Z = Z[None]
    conv = str(d["phase_convention"]) if "phase_convention" in d.files else "igor-raw"
    Z, converted = _to_model_phase(Z, conv, phase)
    if "conditions" in d.files:
        cond = pd.DataFrame(json.loads(str(d["conditions"])))
    elif "bias_V" in d.files:
        cond = pd.DataFrame({"bias_V": np.asarray(d["bias_V"], float)})
    else:
        cond = pd.DataFrame(index=range(Z.shape[0]))
    x = np.asarray(d["x_um"], float)
    o = np.argsort(x)                                  # always ascending in x
    meta = dict(path=str(path), phase_convention_file=conv, phase_converted=converted,
                ref_index=int(d["ref_index"]) if "ref_index" in d.files else None)
    return Series(key or Path(path).stem, probe, x[o], np.asarray(d["freq_Hz"], float),
                  Z[:, o, :], cond, meta)


def load_dense_reference(ds: Dataset, phase: str = "model", axis: bool = True) -> Series:
    """SCM_PIT DenseReference.npz (+ log). Adds the InvOLS-ruler axis ``x_true_um``."""
    probe = PROBES[ds.probe]
    root = config.data_root()
    d = np.load(root / ds.path)
    Z, converted = _to_model_phase(np.asarray(d["Z"]), "igor-raw", phase)
    x = np.asarray(d["x_um"], float)
    meta = dict(path=str(root / ds.path), phase_convention_file="igor-raw",
                phase_converted=converted)
    s = Series(ds.key, probe, x, np.asarray(d["freq_Hz"], float), Z[None],
               pd.DataFrame([{"bias_V": 0.0}]), meta)
    logp = root / ds.extra["log"] if "log" in ds.extra else None
    if logp is not None and logp.exists():
        log = pd.read_csv(logp)
        log["i"] = log.tune_file.str.extract(r"_(\d{4})\.txt$")[0].astype(int)
        s.meta["log"] = log
        if axis:
            from .axis import invols_ruler_axis
            s.meta.update(invols_ruler_axis(log, x, n_anchor=ds.extra.get("n_anchor", 10)))
    return s


def load_folder_series(ds: Dataset, phase: str = "model") -> dict:
    """Folder datasets (load ladder, AC series): {value_from_filename: Series}."""
    probe = PROBES[ds.probe]
    out = {}
    for p in sorted((config.data_root() / ds.path).glob("*_checkpoint.npz")):
        m = re.search(r"_(\d+(?:\.\d+)?)(nN|mV)_checkpoint", p.name)
        val = float(m.group(1)) if m else p.stem
        out[val] = load_series_file(p, probe, f"{ds.key}:{val}", phase)
    return out


def load_json(key_or_path):
    p = path_of(key_or_path) if key_or_path in DATASETS else Path(key_or_path)
    if p.suffix == ".npz":
        d = np.load(p, allow_pickle=True)
        return {k: d[k] for k in d.files}
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def load(key: str, phase: str = "model", **kw):
    """Load a registered dataset by key. Returns Series, dict of Series, or dict."""
    ds = DATASETS[key]
    p = config.data_root() / ds.path
    if not p.exists():
        raise FileNotFoundError(
            f"{key}: {p} not found. Check config.describe() -- data_root may be wrong.")
    if ds.kind == "dense":
        return load_dense_reference(ds, phase=phase, **kw)
    if ds.kind == "series":
        if p.is_dir():
            return load_folder_series(ds, phase)
        return load_series_file(p, PROBES[ds.probe], key, phase)
    if ds.kind == "json":
        return load_json(key)
    if ds.kind == "tune_txt":
        return load_igor_tune_txt(p, phase)
    raise ValueError(ds.kind)


def available() -> pd.DataFrame:
    """Registry as a table with an 'on_disk' column -- first cell of every notebook."""
    root = config.data_root()
    rows = []
    for k, d in DATASETS.items():
        p = root / d.path
        rows.append(dict(key=k, probe=d.probe, kind=d.kind, figure=d.figure,
                         on_disk=p.exists(), description=d.description))
    return pd.DataFrame(rows)


def load_igor_tune_txt(path, phase: str = "model"):
    """Igor text-wave tune export (columns Frequency, Phase[deg], Amp[V]) -> (freq_Hz, Z complex).

    These files are Igor-raw phase; conjugated to the model convention by default.
    """
    t = Path(path).read_bytes().decode("latin-1").replace("\r", "\n").split("\n")
    i = t.index("BEGIN")
    a = np.array([[float(v) for v in ln.split("\t")[1:4]] for ln in t[i + 1:] if ln.startswith("\t")])
    f, ph, amp = a.T
    z = amp * np.exp(1j * np.radians(ph))
    return f, (np.conj(z) if phase == "model" else z)
