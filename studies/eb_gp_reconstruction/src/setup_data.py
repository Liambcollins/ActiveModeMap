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

import numpy as np, pandas as pd, activemodemap as amm
from activemodemap.lowrank import band_mask, resonance_index, classify_null_from_map
from config import DENSE_GRID_A
D=str(DENSE_GRID_A)
BANDS=dict(A=(270e3,450e3), B=(860e3,1300e3))
def load(mode='A', axis='calibrated'):
    dn=np.load(D+'/DenseReference.npz',allow_pickle=True)
    x=dn['x_um'].astype(float); f=dn['freq_Hz']; Z=dn['Z']
    bm=band_mask(f,BANDS[mode]); f=f[bm]; Z=Z[:,bm]
    if axis=='calibrated':
        dl=pd.read_csv(D+'/DenseReference_log.csv',parse_dates=['timestamp'])
        dl['i']=dl.tune_file.str.extract(r'_(\d{4})\.txt$')[0].astype(int)
        ps=amm.fit_position_scale(dl[dl.i<20],dl[dl.i>=20],verbose=False)
        x=ps.apply(x)
    else:
        ps=amm.PositionScale.identity()
    ires=resonance_index(f,Z)
    return x,f,Z,ires,ps
if __name__=='__main__':
    for mode in 'AB':
        for axis in ['nominal','calibrated']:
            x,f,Z,ires,ps=load(mode,axis)
            cl=classify_null_from_map(x,Z,f,ires)
            print(f'mode {mode} {axis:11s}: x {x.min():.2f}-{x.max():.2f}  nf={len(f)}  f_res={f[ires]/1e3:.2f} kHz')
            print(f'   GROUND TRUTH null: status={cl["status"]:17s} x_null={cl["x_null_um"]:.2f}  bound={cl["x_bound_um"]:.2f}  gap={cl["gap_Hz"]/1e3:.1f} kHz  crossings={np.round(cl["crossings_um"],2)}')
        print()
