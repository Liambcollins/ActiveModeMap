"""Central paths for the EB / EB+GP / GP reconstruction study.

Every script in this study resolves data through these variables, so moving a
dataset means editing one file (or setting the environment variable). Defaults
are the paths on the machine the study was developed on.

    AMM_DENSE_GRID_A   the dense reference experiment (101 positions)
    AMM_BIAS_SERIES    the bias-dependence series checkpoint
    AMM_EB_REPO        EB-Solver-CResonance clone (the forward model)
    AMM_FEM_LADDER     AFeMulator ladder exports  (archive/fem only)
    AMM_STUDY_OUT      where libraries, CSVs and figures are written

Nothing here reads the data; importing config.py is free and safe.
"""
import os
from pathlib import Path

HERE = Path(__file__).resolve().parent


def _p(env, default):
    return Path(os.environ.get(env, default))


# Root of the experiment tree.  Note the default is a WINDOWS path, and this
# file must still behave when it is read by a POSIX interpreter (a WSL shell, a
# container with the folder mounted).  Under PurePosixPath a backslash is an
# ordinary character, so `Path(r'D:\...\Dense_Grid_A').parent` is '.', not the
# SCM_PIT directory -- deriving one dataset from another's `.parent` silently
# produces a bare relative path.  So each location is rooted at SCM_PIT and only
# ever EXTENDED (which is safe: Windows accepts mixed separators).
SCM_PIT = _p('AMM_SCM_PIT', r'D:\ActiveModeMap\SCM_PIT')

# The dense reference experiment (ground truth for the retrospective replay)
DENSE_GRID_A = _p('AMM_DENSE_GRID_A', SCM_PIT / 'Dense_Grid_A')

# The bias-dependence series (src/bias.py)
BIAS_SERIES = _p('AMM_BIAS_SERIES',
                 SCM_PIT / 'Bias Dependence' / 'series_checkpoint.npz')

# EB solver repo (external dependency).  Two importable directories, and they
# are NOT both under src/: eb_cr_afm is in src/, while eb_models.py and
# geometry.py live in fem_eb_comparison/py/.  Getting this wrong is what made an
# earlier pass conclude the EB library could not be rebuilt.
EB_REPO = _p('AMM_EB_REPO',
             r'C:\Users\lz1\Documents\Github\EB-Solver-CResonance')
EB_SRC = EB_REPO / 'src'
EB_PY = EB_REPO / 'fem_eb_comparison' / 'py'
EB_GEOMETRY = EB_REPO / 'fem_eb_comparison' / 'results' / 'geometry_Multi75G.json'

# FEM ladder exports + manifest (AFeMulator output).  Only archive/fem/ needs
# this; the three headline arms do not.
FEM_LADDER = _p('AMM_FEM_LADDER', EB_REPO / 'fem_eb_comparison' / 'data_ladder')

# Where libraries, benchmark CSVs and figures are written
OUT = _p('AMM_STUDY_OUT', HERE / 'results')
FIG = OUT / 'fig'

# The surrogate libraries, by the name each builder writes.  Sizes are the
# uncompressed footprint on disk; none of these is committed.
LIBRARIES = {
    'eb':      OUT / 'eblib.npz',              # buildlib.py, log-amp only
    'eb_cplx': OUT / 'eblib_cplx.npz',         # buildlib_cplx.py, 185-720 kHz
    'eb_x':    OUT / 'eblib_x_cplx.npz',       # buildlib_x_cplx.py, 870 MB
    'fem':     OUT / 'femlib_isotropic_cplx.npz',   # archive/fem
}


def add_eb_to_path():
    """Put BOTH EB directories on sys.path (see EB_PY above)."""
    import sys
    for p in (str(EB_SRC), str(EB_PY)):
        if p not in sys.path:
            sys.path.insert(0, p)
