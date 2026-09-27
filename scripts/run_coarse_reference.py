r"""Same-session coarse reference sweep -- 5 um pitch, after the EB+GP campaign.

Why: the 1 um dense map is ~16 h old and the drift check (19:49-19:53) showed
CR1 wandering by -70..+117 Hz between positions, which on a Q~130 line puts a
0.2-0.35 floor on complex NRMSE against that map. A reference taken in the same
session as the campaign removes that floor from the comparison.

Same code path as the dense map (`run_series` with a fixed design, see
dense_map_100nN_0V_log.txt: "fixed design ... no convergence stop"), same
condition (100 nN, 0 V, spot 1), same direction (445 -> 100), 5 um pitch:
70 positions, ~70 s each, ~85 min. Checkpointed every position; resumable.

RUN FROM THE KERNEL THAT HOLDS `inst`, on the main thread, AFTER the campaign:

    %run -i "C:/Users/Asylum User/Desktop/STAFF Software/Liam/ActiveModeMap/scripts/run_coarse_reference.py"
"""
import os
import sys
import time

import numpy as np

from activemodemap.series import make_conditions, run_series

OUT = r'D:\User Data\Liam\ActiveModeMap\DomainsBPPPCONTAU'
CKPT = os.path.join(OUT, 'coarse_ref_5um_100nN_0V_checkpoint.npz')
LOG = os.path.join(OUT, 'coarse_ref_5um_100nN_0V_log.txt')

PITCH_UM = 5.0
X_FIXED = np.arange(445.0, 100.0 - 1e-9, -PITCH_UM)   # 445 -> 100 inclusive, 70 positions
LOAD_NN, BIAS_V, SPOT = 100.0, 0.0, 1


class Tee:
    def __init__(self, path):
        self.f = open(path, 'a', buffering=1)
        self.out = sys.stdout          # the notebook stream, not sys.__stdout__

    def write(self, s):
        self.out.write(s); self.f.write(s)

    def flush(self):
        self.out.flush(); self.f.flush()


def main(inst):
    cond = make_conditions([BIAS_V], [LOAD_NN], spots=[SPOT])
    print(f"\n=== COARSE REFERENCE started {time.strftime('%Y-%m-%d %H:%M:%S')} ===")
    print(f"{X_FIXED.size} positions {X_FIXED[0]:.0f} -> {X_FIXED[-1]:.0f} um at {PITCH_UM:.0f} um, "
          f"{LOAD_NN:.0f} nN, {BIAS_V:+.1f} V, spot {SPOT}")
    print(f"instrument before: x={inst.current_x} um, spot={getattr(inst, 'current_spot', None)}")
    print(f"checkpoint: {CKPT}")
    res = None
    try:
        res = run_series(inst, X_FIXED, cond, positions_um=X_FIXED,
                         checkpoint_path=CKPT, resume=True, verbose=True)
    finally:
        try:
            inst.set_dc_bias(0.0)
        except Exception as e:
            print('bias reset:', e)
        try:
            inst.a.withdraw()
        except Exception as e:
            print('withdraw:', e)
        print(f"=== COARSE REFERENCE finished {time.strftime('%Y-%m-%d %H:%M:%S')} "
              f"(tip withdrawn, bias 0 V) ===")
    if res is not None:
        print(f"coarse reference: {len(res['x_um'])} positions x {len(res['freq_Hz'])} frequencies")
    return res


if __name__ == '__main__':
    if 'inst' not in globals():
        raise SystemExit("No `inst` in the namespace -- run with `%run -i` from the kernel that holds it.")
    _tee = Tee(LOG); _old = sys.stdout; sys.stdout = _tee
    try:
        result = main(globals()['inst'])
    finally:
        sys.stdout = _old; _tee.flush()
