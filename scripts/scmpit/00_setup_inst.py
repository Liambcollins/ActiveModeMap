r"""Build (or adopt) `inst` for the SCM-PIT campaign on DomainsB.

- Reuses an existing `inst` if one is in the namespace (position bookkeeping is
  software-tracked, so a second instrument must never be built blind).
- Otherwise connects to Igor and builds automation + inst with the detection
  spot assumed AT THE FREE END (Liam, 2026-09-20: "the detection spot is
  currently at the free end").
- Sets Igor's save folder to DomainsB_SCMPIT and base filename to SCMPIT.

Span: x = 0 is the clamped base. PROBE_L_UM starts as the earlier SCM-PIT value
(232 um) and is refined by 01_preflight.py; x_limits are wide open (0, L) here
because the pre-flight is what measures the reachable range. Only the pre-flight
walks toward the base -- every later stage uses the measured limits.
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scmpit_common as C
import numpy as np

from activemodemap.asylum import AFMLaserSweepAutomation, AsylumInstrument

os.makedirs(C.FILE_LOC, exist_ok=True)
FORCE_NEW = globals().get('FORCE_NEW_INST', False)

if 'inst' in globals() and not FORCE_NEW:
    inst = globals()['inst']
    C.stamp(f'adopting existing inst: x={inst.current_x} um, spot={inst.current_spot}, '
            f'load={inst.load_nN} nN, bias={inst.dc_bias_V:+.1f} V, span={inst.span_um}, '
            f'limits={inst.x_limits_um}')
    if inst.a.file_loc != C.FILE_LOC or inst.a.base_filename != C.BASE_FILENAME:
        inst.a.file_loc = C.FILE_LOC
        inst.a.base_filename = C.BASE_FILENAME
        inst.a.set_folder()
        C.stamp(f'repointed save folder -> {C.FILE_LOC}, base filename -> {C.BASE_FILENAME}')
else:
    import win32com.client
    igor = win32com.client.Dispatch('IgorPro.Application')
    C.stamp('connected to Igor Pro')

    automation = AFMLaserSweepAutomation(igor, C.FILE_LOC, C.BASE_FILENAME,
                                         log_filename=os.path.join(C.FILE_LOC, 'SCMPIT_log'))
    automation.eigenmode_center_freq = 275e3      # CR1 guess only; the tune window is the panel's
    automation.resonance_band_Hz = None           # report f_res over the whole (2 MHz) window
    automation.autowedge_pause = 15.0
    automation.wait_for_tune_complete = True
    automation.tune_timeout_s = 600.0             # 10 s tunes + margin
    automation.verbose_tune = True
    automation.spot_timeout_s = 90.0
    automation.spot_stable_tol_V = 2e-3
    automation.set_folder()
    C.stamp(f'save folder -> {C.FILE_LOC}, base filename -> {C.BASE_FILENAME}')

    L = C.PROBE_L_UM_FALLBACK
    inst = AsylumInstrument(automation, load_nN=500.0, dc_bias_V=0.0,
                            recalibrate_each=True, analysis_band_Hz=None,
                            start_at='free_end', span_um=L, x_limits_um=(0.0, L),
                            hw_sign_toward_free_end=C.HW_SIGN_TOWARD_FREE_END,
                            apply_bias=False)
    inst.set_position(L, confirm=True)            # spot IS at the free end (Liam, 11:xx)
    C.stamp(f'inst built: laser bookkeeping x = {inst.current_x} um (free end of a {L:.0f} um lever, provisional)')
    inst.preview_moves([L, L - 10.0])             # second entry must read "toward base"

gmv = inst.a.get_gmv()
k_panel = gmv.get('SpringConstant')
C.stamp(f'MasterPanel: k = {k_panel} N/m (campaign {C.K_LEVER_N_PER_M}), '
        f'InvOLS = {gmv.get("InvOLS")}, setpoint = {gmv.get("DeflectionSetpointVolts")} V')
if k_panel and abs(k_panel - C.K_LEVER_N_PER_M) / C.K_LEVER_N_PER_M > 0.10:
    print('  *** panel spring constant differs from the thermal 1.697 N/m by >10 % -- '
          'check the MasterPanel before trusting any load in newtons ***')
C.stamp('00_setup_inst done')
