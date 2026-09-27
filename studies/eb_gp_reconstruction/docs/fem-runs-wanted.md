# FEM runs wanted (all optional except the first)

Everything below runs against the existing AFeMulator setup; commands assume
`fem_eb_comparison\py\` as the working directory with the app open in a
visible browser tab.

## 1. Frictionless library (no new FEM — one stitch command, ~2 min)

Resolves the Q6 lateral-spring confound. The 160 exports already exist.

    python build_fem_library.py --data ..\data_ladder --name Multi75G ^
        --lateral frictionless --out femlib_frictionless.npz

Then benchmark it (from studies/physics_informed_al/):

    python src\run_fem.py results\femlib_frictionless.npz 24 fem_fl results\femfl_bench.csv

## 2. e_tip mini-ladder (12 exports, ~3-25 min)

Tests whether the elastic tip column is what caps FEM's f2/f1 at 2.66-2.94
against the measured 3.080. Copy `libraries/run_fem_etip.py` next to
`run_fem_ladder.py` and run:

    python run_fem_etip.py --stl <ABSOLUTE STL PATH> --out ..\data_etip

Reading: if f2/f1 climbs toward ~3.1 as e_tip rises, the tip column explains
the saturation (and e_tip becomes a calibratable knob, the FEM analogue of the
EB cone stiffness). If it barely moves, look at the taper/overhang/attach node.

## 3. Later, if the two-band fit pans out

Nothing new — the ladder exports already span 0-2 MHz. Rebuilding the FEM
library with `U_HI = 3.5` in build_fem_library.py is a local re-stitch.
