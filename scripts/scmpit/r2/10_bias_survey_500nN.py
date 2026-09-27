r"""Stage 1 -- two-domain bias survey at 500 nN on the SCM-PIT probe.

Same design as bias_domain_survey_250nN.py (and the 15 nN survey) so the runs
are directly comparable: 8 equispaced positions over the REACHABLE span, bias
[-9,-6,-3,0,3,6,9] V interleaved, spots 1 and 2 as the outermost loop, free end
first. 14 conditions x 8 positions ~ 40 min.

Prediction being tested (handoff, 2026-09-20): this probe is 4.2x stiffer than the
PPP-CONTAu and runs at 500 nN, so the channel ratio |b|/|P| should come out
markedly below every PPP-CONTAu value (CR1 1.64 at 15 nN), while V_cpd (~+1.5 V
on PPP-CONTAu) should be recovered independent of mode. Analysis is a separate
pass (separate_domains / separate_channels / channel_spots / estimate_v_cpd).

Everything position/frequency-dependent comes from preflight_result.json.
"""
import os
import sys
import time
import contextlib

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import scmpit_common as C
# --- R2 guard -------------------------------------------------------------
# Byte-identical to the R1 script apart from this block. Refuses to run unless
# 00_setup_r2.py has repointed the campaign at the R2 data folder, so a failed
# setup aborts the queue instead of writing a second campaign into R1's folders.
if not C.FILE_LOC.rstrip('\\/').endswith('_R2'):
    raise SystemExit(f'R2 guard: C.FILE_LOC is {C.FILE_LOC!r} -- run 00_setup_r2.py first')
from activemodemap import make_conditions, run_series, load_checkpoint

inst = globals().get('inst')
P = C.load_preflight()
C.check_inst(inst, P)

PROBE_L_UM, X_LO_UM = float(P['probe_L_um']), float(P['x_lo_um'])
BIAS_V = [-9, -6, -3, 0, 3, 6, 9]
BIAS_ORDER = 'interleaved'
SPOT_UP, SPOT_DOWN = 1, 2
LOAD_NN = [500.0]
REF_INDEX = 0
RANK = 4
DNS_BAND_HZ = C.cr1_band(P)
SPARSE_X_UM = np.round(np.linspace(X_LO_UM, PROBE_L_UM, 8), 1)
SPARSE_ORDER = SPARSE_X_UM[::-1]                       # free end first
CHECKPOINT = os.path.join(C.FILE_LOC, 'domains_bias_checkpoint_500nN_scmpit.npz')
LOG = os.path.join(C.FILE_LOC, 'bias_domain_survey_500nN_scmpit_log.txt')

_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
t0 = time.time()
series = None
try:
    C.stamp('=' * 78)
    C.stamp(f'STAGE 1  TWO-DOMAIN BIAS SURVEY AT {LOAD_NN[0]:.0f} nN  (SCM-PIT, k = {C.K_LEVER_N_PER_M} N/m, '
            f'deflection {LOAD_NN[0] * 1e-9 / C.K_LEVER_N_PER_M * 1e9:.0f} nm)')
    C.stamp(f'CR1 {P["cr1_Hz"] / 1e3:.2f} kHz -> analysis band {DNS_BAND_HZ[0] / 1e3:.0f}-{DNS_BAND_HZ[1] / 1e3:.0f} kHz')
    C.stamp('=' * 78)
    conditions = make_conditions(BIAS_V, LOAD_NN, spots=[SPOT_UP, SPOT_DOWN], bias_order=BIAS_ORDER)
    C.stamp(f'{len(conditions)} conditions/position: ' + ', '.join(str(c) for c in conditions))
    C.stamp(f'positions: {SPARSE_X_UM.tolist()}  (order: {SPARSE_ORDER.tolist()})')
    C.stamp(f'instrument before: x={inst.current_x} um, spot={inst.current_spot}')
    C.stamp(f'checkpoint: {CHECKPOINT}')
    series = run_series(inst, SPARSE_X_UM, conditions, ref_index=REF_INDEX, rank=RANK,
                        dns_band_Hz=DNS_BAND_HZ, checkpoint_path=CHECKPOINT, resume=True,
                        positions_um=SPARSE_ORDER, verbose=True)
    C.stamp(f"{series['x_um'].size} positions x {len(conditions)} conditions x "
            f"{series['freq_Hz'].size} frequencies -- survey done")
    # quick sanity: CR1 at 0 V, spot 1, mean over positions
    F, Z = series['freq_Hz'], series['Z']
    band = (F > DNS_BAND_HZ[0]) & (F < DNS_BAND_HZ[1])
    f_cr1 = F[band][np.argmax(np.abs(Z[REF_INDEX, :, band]).mean(axis=0))]
    C.stamp(f'CR1 (0 V, spot {SPOT_UP}, mean over positions): {f_cr1 / 1e3:.2f} kHz '
            f'(pre-flight free end {P["cr1_Hz"] / 1e3:.2f} kHz)')
except Exception as e:
    C.stamp(f'*** FAILED: {type(e).__name__}: {e}')
    import traceback; traceback.print_exc()
    raise
finally:
    C.finish(inst, t0, 'STAGE 1 BIAS SURVEY')
    sys.stdout = _old; _tee.close()

stage1_series = series
