"""Paths and environment for the paper analysis.

Zero-config by default: this folder is expected to live at
``D:\\ActiveModeMap\\paper_analysis``, so the data root is its parent folder.
Override any path with an environment variable or a ``config.local.json``
next to this package's parent folder::

    {"data_root": "D:/ActiveModeMap",
     "amm_repo": "C:/Users/lz1/Documents/Github/ActiveModeMap_repo/ActiveModeMap",
     "eb_repo": "C:/Users/lz1/Documents/Github/EB-Solver-CResonance"}

Environment variables (take precedence): FMM_DATA_ROOT, FMM_AMM_REPO, FMM_EB_REPO.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

PKG_DIR = Path(__file__).resolve().parent            # .../paper_analysis/fmmpaper
PROJECT_DIR = PKG_DIR.parent                          # .../paper_analysis
FIG_DIR = PROJECT_DIR / "figures"
RESULTS_DIR = PROJECT_DIR / "results"

_DEFAULTS = {
    "data_root": str(PROJECT_DIR.parent),
    "amm_repo": r"C:\Users\lz1\Documents\Github\ActiveModeMap_repo\ActiveModeMap",
    "eb_repo": r"C:\Users\lz1\Documents\Github\EB-Solver-CResonance",
    "fmm_code_zip": r"C:\Users\lz1\Documents\Github\ActiveModeMap_repo\Manuscript_FMM\FMM_analysis_code.zip",
}
_ENV = {"data_root": "FMM_DATA_ROOT", "amm_repo": "FMM_AMM_REPO", "eb_repo": "FMM_EB_REPO",
        "fmm_code_zip": "FMM_CODE_ZIP"}


def _load_local() -> dict:
    p = PROJECT_DIR / "config.local.json"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            return json.load(fh)
    return {}


def get(key: str) -> Path:
    """Resolve a configured path: env var > config.local.json > default."""
    if key not in _DEFAULTS:
        raise KeyError(f"unknown config key {key!r}; valid: {sorted(_DEFAULTS)}")
    val = os.environ.get(_ENV[key]) or _load_local().get(key) or _DEFAULTS[key]
    return Path(val)


def data_root() -> Path:
    return get("data_root")


def ensure_activemodemap(require_phase_fix: bool = True) -> None:
    """Make ``import activemodemap`` work, adding the repo to sys.path if needed.

    With ``require_phase_fix`` (default) the package must be the 2026-09-17+ version
    (``asylum.PHASE_SIGN``, ``domains``, fitted zeta/setback): branch
    ``feat/position-scale-and-reanalysis`` or later. GitHub main 580c491 is too old.
    """
    try:
        import activemodemap  # noqa: F401
    except ImportError:
        repo = get("amm_repo")
        if not (repo / "activemodemap").is_dir():
            raise ImportError(
                "activemodemap not importable and not found at "
                f"{repo}. `pip install -e <ActiveModeMap repo>` or set FMM_AMM_REPO.")
        sys.path.insert(0, str(repo))
        import activemodemap  # noqa: F401
    if require_phase_fix:
        import importlib.util
        from activemodemap import asylum
        if not hasattr(asylum, "PHASE_SIGN") or importlib.util.find_spec("activemodemap.domains") is None:
            raise ImportError(
                f"activemodemap at {activemodemap.__file__} predates the 2026-09-17 phase fix. "
                "Check out branch feat/position-scale-and-reanalysis (or a later main).")


def describe() -> str:
    lines = [f"project : {PROJECT_DIR}"]
    for k in _DEFAULTS:
        p = get(k)
        lines.append(f"{k:9s}: {p}  ({'ok' if p.exists() else 'MISSING'})")
    return "\n".join(lines)


FIG_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)
