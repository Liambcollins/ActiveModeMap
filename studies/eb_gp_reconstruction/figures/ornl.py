"""ORNL brand styling for deck figures.

Colors taken verbatim from the theme of the user's ORNL template
(ppt/theme/theme1.xml, clrScheme "ORNL Color TEST"), except ORANGE, which is the
brand accent5 #FF9E1B stepped one notch darker to clear the 3:1 contrast check
against a white slide -- the brand value fails both the lightness band and
contrast.  The trio (GREEN, BLUE, ORANGE) passes all six checks of the dataviz
validator in light mode.
"""
# Paths are resolved through config.py -- set AMM_* environment
# variables or edit that file to point at your data.
import sys as _sys, os as _os
_D = _os.path.dirname(_os.path.abspath(__file__))
while not _os.path.exists(_os.path.join(_D, 'config.py')):
    _D = _os.path.dirname(_D)
_sys.path[:0] = [_D, _os.path.join(_D, 'src'),
                 _os.path.join(_D, 'figures')]
from config import DENSE_GRID_A, FEM_LADDER, EB_GEOMETRY, OUT, FIG, add_eb_to_path
add_eb_to_path()

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

GREEN = '#00662C'   # dk2  -- the ORNL signature green
BLUE = '#006BA6'    # accent2
TEAL = '#00454D'    # accent1
MINT = '#00B38F'    # accent3
LIME = '#7DBA00'    # accent4
ORANGE = '#C87A00'  # accent5 #FF9E1B, darkened to pass contrast
MAGENTA = '#B50093'  # accent6
INK = '#373A36'     # dk1
GREY = '#DBDCDB'    # lt2
WHITE = '#FFFFFF'
MUT = '#7A7D78'
INK2 = '#5A5D58'

# hue = forward model
HUE = {'fem': GREEN, 'eb': BLUE, 'lowrank': ORANGE}
LBL = {'lowrank': '3  low-rank', 'eb': '1  EB', 'eb_gp': '2  EB + GP',
       'fem': '1b  FEM', 'fem_gp': '1b  FEM + GP'}
MK = {'lowrank': 'o', 'eb': 's', 'eb_gp': 's', 'fem': 'D', 'fem_gp': 'D'}


def style():
    plt.rcParams.update({
        'figure.facecolor': WHITE, 'axes.facecolor': WHITE,
        'savefig.facecolor': WHITE,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans'],
        'text.color': INK, 'axes.labelcolor': INK,
        'xtick.color': INK2, 'ytick.color': INK2,
        'axes.edgecolor': MUT, 'axes.linewidth': .8,
        'grid.color': '#ECEDEB', 'grid.linewidth': .8,
        'axes.grid': True, 'axes.axisbelow': True,
        'legend.frameon': False,
        'xtick.labelsize': 9.5, 'ytick.labelsize': 9.5,
        'axes.labelsize': 10.5, 'axes.titlesize': 11.5,
        'figure.dpi': 170,
    })


def clean(ax, grid=True):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(grid)
    ax.tick_params(length=3, width=.8)
    return ax


def title(ax, t):
    ax.set_title(t, loc='left', fontweight='600', pad=8, color=INK)
