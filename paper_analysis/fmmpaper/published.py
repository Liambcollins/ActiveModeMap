"""Read the draft's archived benchmark tables straight from FMM_analysis_code.zip.

The draft's EB+GP arm needs the two-segment EB library binding (several GB), so
it is not re-run by these notebooks. Instead the archived result tables are read
in place, so notebooks can overlay published EB/EB+GP/oracle numbers on freshly
recomputed GP and low-rank curves.

Tables (pandas DataFrames, one row per reconstruction):
  regen       arms gp / eb / eb_gp / ebgp3 x strategies x N  (draft Figs 2-5)
  bench_C_gp  GP arm, all strategies incl. 50 random trials   (draft Fig. 4, S3)
  oracle      hindsight-oracle subsets; objective 'global' is the FULL-band metric
  dns2        D-NS criteria (branch / strict) per reconstruction (draft S4)

Caveat: the zip is dated 2026-08-21. The draft ver091626 quotes a mode-A-band
oracle (36.2, 9.3, 5.9, 5.7, 5.6, 5.1 %) that is not in this archive's oracle.pkl.
"""
from __future__ import annotations

import io as _io
import pickle
import zipfile

from . import config

TABLES = ("regen", "bench_C_gp", "bench_L_gp", "bench_C_eb", "oracle", "dns2", "ampstudy")


def load(name: str):
    if name not in TABLES:
        raise KeyError(f"{name!r}; available: {TABLES}")
    zp = config.get("fmm_code_zip")
    if not zp.exists():
        raise FileNotFoundError(f"{zp} not found (set FMM_CODE_ZIP or config.local.json)")
    with zipfile.ZipFile(zp) as z:
        return pickle.load(_io.BytesIO(z.read(f"out/{name}.pkl")))
