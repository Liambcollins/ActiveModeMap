"""Slide figure: the two forward models, side by side.

Left  -- two-segment Euler-Bernoulli with a contact spring (analytic).
Right -- the AFeMulator model drawn from the ACTUAL geometry in a ladder export:
         STL silhouette in side elevation plus the beam-network nodes, so the
         picture is the model that produced the library, not an impression of it.
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

import numpy as np, h5py, matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
import ornl as O

H = (str(FEM_LADDER) +
     '/fem_Multi75G_frictionless_mech_k1_1248_ld4.78.h5')

O.style()
fig = plt.figure(figsize=(13.2, 4.3))
a1 = fig.add_axes([.035, .13, .445, .755])
a2 = fig.add_axes([.545, .13, .445, .755])

# ---------------- left: two-segment EB -------------------------------------
a1.set_axis_off()
L, xc, xn = 1.0, .885, .958            # contact at eta 1 -> .885; node past it
a1.plot([0, L], [0, 0], color=O.BLUE, lw=5.5, solid_capstyle='butt', zorder=3)
a1.add_patch(plt.Rectangle((-.05, -.115), .05, .23, color=O.INK, zorder=4))
for y in np.linspace(-.115, .115, 7):
    a1.plot([-.068, -.05], [y, y + .030], color=O.INK, lw=.9, zorder=4)
a1.plot([xc, xc], [-.02, -.17], color=O.GREEN, lw=1.6, zorder=3)
zz = np.linspace(-.17, -.30, 60)
a1.plot(xc + .022 * np.sin(np.linspace(0, 7 * np.pi, 60)), zz,
        color=O.GREEN, lw=1.8, zorder=3)
a1.plot([xc - .06, xc + .06], [-.315, -.315], color=O.INK, lw=2.2, zorder=3)
for s in np.linspace(-.05, .05, 5):
    a1.plot([xc + s, xc + s - .026], [-.315, -.375], color=O.INK, lw=.9,
            zorder=3)
a1.text(xc, -.425, '$k_1$   contact stiffness', color=O.GREEN, ha='center',
        fontsize=10.5, va='center', fontweight='600')

s = np.linspace(0, L, 400)             # first mode, node deliberately past xc
w = s ** 2 * (xn - s)
w = .155 * w / np.abs(w).max()
a1.plot(s, w, color=O.BLUE, lw=1.7, ls='--', alpha=.7, zorder=2)
a1.plot([xn], [0], 'o', ms=9.5, mfc=O.WHITE, mec=O.MAGENTA, mew=2.3, zorder=6)
a1.annotate('detection node spot (D-NS)\ndetected amplitude vanishes here',
            xy=(xn, .004), xytext=(.50, .265), fontsize=9.8, color=O.MAGENTA,
            ha='left', va='bottom',
            arrowprops=dict(arrowstyle='-', color=O.MAGENTA, lw=1.1,
                            shrinkA=2, shrinkB=4))
a1.annotate('', xy=(0, -.055), xytext=(xc, -.055),
            arrowprops=dict(arrowstyle='<->', color=O.INK2, lw=1.0))
a1.text(xc / 2, -.098, 'segment 1', ha='center', fontsize=9.5, color=O.INK2)
a1.text((xc + L) / 2 + .01, -.098, 'seg 2', ha='center', fontsize=9.5,
        color=O.INK2)
a1.text(-.05, .50, '1   Two-segment Euler–Bernoulli', fontsize=13,
        fontweight='600', color=O.BLUE)
a1.text(-.05, .415, 'Analytic beam, normal contact spring. Fitted: $k_1$, '
                    '$f_{res}$, air damping.', fontsize=10, color=O.INK2)
a1.text(-.05, -.53, 'HAS', fontsize=9.4, color=O.GREEN, fontweight='600',
        va='top')
a1.text(.055, -.53,
        '11° tilt · 13.5 µm tip height as a moment arm · genuine 15.8 µm\n'
        'free overhang ($z\'\' = z\'\'\' = 0$ at the end) · trapezoidal section',
        fontsize=9.4, color=O.INK2, va='top')
a1.text(-.05, -.685, 'LACKS', fontsize=9.4, color=O.BLUE, fontweight='600',
        va='top')
a1.text(.075, -.685,
        'lateral contact spring ($k_2 = 0$) · contact dashpots ($c_1 = c_2 = 0$)\n'
        'tip-cone compliance (rigid) · axial taper · flexural modes only',
        fontsize=9.4, color=O.INK2, va='top')
a1.set_xlim(-.09, 1.11); a1.set_ylim(-.90, .58)

# ---------------- right: the real FEM geometry ------------------------------
with h5py.File(H, 'r') as f:
    V = f['input/geometry/mod_vertices'][:]
    F = f['input/geometry/mod_faces'][:]
    cn = f['mesh/cant_nodes_um'][:]
    tn = f['mesh/tip_nodes_um'][:]
    at = dict(f['mesh'].attrs)
    g = dict(f['input/geometry'].attrs)
Z_EX = 5.0                             # vertical exaggeration, stated on the plot
polys = [[(V[i, 0], V[i, 2] * Z_EX) for i in tri] for tri in F]
a2.add_collection(PolyCollection(polys, facecolors=O.GREY, edgecolors='#BFC1BF',
                                 linewidths=.4, alpha=.95, zorder=1))
a2.plot(cn[:, 0], cn[:, 2] * Z_EX, '-', color=O.GREEN, lw=1.5, zorder=3)
a2.plot(cn[:, 0], cn[:, 2] * Z_EX, 'o', ms=2.6, color=O.GREEN, zorder=4)
a2.plot(tn[:, 0], tn[:, 2] * Z_EX, '-', color=O.MINT, lw=2.0, zorder=3)
a2.plot(tn[:, 0], tn[:, 2] * Z_EX, 'o', ms=2.6, color=O.MINT, zorder=4)
ai = int(at['attach_index'])
a2.plot([cn[ai, 0]], [cn[ai, 2] * Z_EX], 'o', ms=9, mfc=O.WHITE, mec=O.INK,
        mew=2, zorder=6)
a2.plot([0], [0], 'v', ms=9, color=O.MINT, zorder=6)

a2.annotate(f'tip attaches at node {ai} of {len(cn)},\n'
            f'leaving a {V[:, 0].max():.1f} µm overhang',
            xy=(cn[ai, 0], cn[ai, 2] * Z_EX), xytext=(-228, 52),
            fontsize=9.6, color=O.INK, ha='left',
            arrowprops=dict(arrowstyle='-', color=O.INK, lw=1.1))
a2.annotate(f'{len(tn)}-node elastic tip column, {at["L_tip_um"]:.1f} µm —\n'
            'the tip is not a rigid point mass',
            xy=(0, 7.5 * Z_EX), xytext=(-228, 22), fontsize=9.4,
            color='#00786A', ha='left',
            arrowprops=dict(arrowstyle='-', color=O.MINT, lw=1.1))
a2.annotate(f'{len(cn)} beam nodes over {at["L_main_um"]:.1f} µm, '
            f'{g["beam_node_spacing_um"]:.1f} µm spacing',
            xy=(-150, 15.9 * Z_EX), xytext=(-228, 90), fontsize=9.4,
            color=O.GREEN, ha='left',
            arrowprops=dict(arrowstyle='-', color=O.GREEN, lw=1.0))
a2.text(-228, 122, '1b   AFeMulator beam-network FEM', fontsize=13,
        fontweight='600', color=O.GREEN)
a2.text(-228, 105, 'Geometry from the STL: taper, tilt, tip set-back, overhang.',
        fontsize=10, color=O.INK2)
a2.text(-228, -16,
        'Flexural, lateral and torsional families. Same three fitted\n'
        'parameters — the added realism is fixed geometry, not more knobs.',
        fontsize=9.4, color=O.INK2, va='top')
a2.text(-228, -52,
        'The isotropic library also sets $k_x = k_y = k_z$, so it HAS the\n'
        'lateral contact spring EB lacks — a second difference, not just '
        'geometry.', fontsize=9.4, color=O.BLUE, va='top')
a2.text(20, -99, f'side elevation, z × {Z_EX:.0f}   ', fontsize=8.8,
        color=O.MUT, ha='right', style='italic')
a2.set_xlabel('x from contact point  (µm)', fontsize=10)
a2.set_xlim(-234, 26); a2.set_ylim(-104, 132)
a2.set_yticks([])
O.clean(a2, grid=False)
for sp in ('left', 'right', 'top'):
    a2.spines[sp].set_visible(False)

fig.savefig(str(FIG) + '/d_models.png', dpi=170)
print('wrote fig/d_models.png')
