"""fmmpaper -- shared analysis layer for the Fast Mode Mapping / quantitative CR-PFM paper.

Modules
-------
config    paths (data root, activemodemap repo, EB repo); zero-config on Liam's laptop
registry  probes and datasets by key -- the only place file paths are written
io        loaders -> Series (Z[condition, position, freq], model phase convention)
axis      InvOLS position ruler, pre-flight InvOLS interpolation, static-shape fit
spectra   peaks, Q, mode profiles, nodes, detection null
domains   two-domain bias decomposition, V_cpd, enhancement E(x), d33
recon     GP / low-rank reconstruction, selectors, held-out and cross-capture scoring
plotting  one figure style; save to figures/ as PNG + PDF
results   save/load derived numbers (results/*.json, *.csv) with provenance
"""
from . import config, registry  # noqa: F401
from .io import Series, available, load  # noqa: F401

__version__ = "0.1.0"
