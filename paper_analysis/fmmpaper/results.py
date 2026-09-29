"""Write and read derived numbers so text, tables and figures never retype a value.

    from fmmpaper import results
    results.save("fig6_transfer", {"d33_qs_median": 8.81, ...}, table=df)
    results.load("fig6_transfer")["d33_qs_median"]

Each save writes results/<name>.json (numbers + provenance) and, if given,
results/<name>.csv (the table behind a figure).
"""
from __future__ import annotations

import datetime as _dt
import json
import platform

import numpy as np

from . import config


def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, complex):
        return [o.real, o.imag]
    return str(o)


def provenance() -> dict:
    import scipy
    return dict(created=_dt.datetime.now().isoformat(timespec="seconds"),
                python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
                data_root=str(config.data_root()))


def save(name: str, values: dict, table=None):
    payload = dict(values=values, provenance=provenance())
    p = config.RESULTS_DIR / f"{name}.json"
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1, default=_default)
    if table is not None:
        table.to_csv(config.RESULTS_DIR / f"{name}.csv", index=False)
    return p


def load(name: str) -> dict:
    with open(config.RESULTS_DIR / f"{name}.json", encoding="utf-8") as fh:
        return json.load(fh)["values"]
