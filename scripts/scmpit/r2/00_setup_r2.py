r"""R2 SETUP -- repoint the whole campaign at a fresh data folder.

Run this ONCE, before any R2 stage. It mutates the already-imported
`scmpit_common` module so every later script -- including the unmodified R1
stage scripts -- writes into the new folder instead of the original one. Igor's
own save paths (images, force curves, tune text files) are repointed too, via
`inst.a.file_loc` + `set_folder()`.

Nothing is moved or deleted. `DomainsB_SCMPIT` (R1, 2026-09-20) is left exactly
as it was reorganised; R2 starts empty.

Every R2 script carries a guard that refuses to run unless `C.FILE_LOC` ends in
`_R2`, so if this script fails the rest of the queue aborts harmlessly instead of
writing a second campaign's worth of files into R1's folders.

Two deliberate changes from R1, both logged:

  * `invols_bounds` ceiling 1e-5 -> 5e-5 m/V. The R1 close-out measured
    1.993e-5 m/V at the base position (52 um) and the old ceiling silently
    refused to write it to the MasterPanel, leaving Igor's own deflection
    calibration stale there. The bound is a sanity check, not a calibration, and
    at this tip's grown InvOLS it was cutting into real values. This does not
    change any spectrum we analyse (the load setpoint uses the returned value
    either way) -- only what Igor stores.
  * A prior V_cpd is seeded into `vcpd_current.json` from the R1 close-out so
    that stage 5 has a sane second bias even if the stage-1 recomputation fails.
    It is overwritten by `11_vcpd_from_stage1.py` and again by
    `55_vcpd_from_images.py`, each time with a fresher, R2-measured number.
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import scmpit_common as C

inst = globals().get('inst')
C.check_inst(inst)

R2 = r'D:\User Data\Liam\ActiveModeMap\DomainsB_SCMPIT_R2'
R1_CLOSEOUT_VCPD = 1.191          # quasi-static band, 80_closeout_cal.py, 22:19
OLD_FILE_LOC = C.FILE_LOC

os.makedirs(R2, exist_ok=True)

# --- repoint the campaign ----------------------------------------------------
C.FILE_LOC = R2
C.PREFLIGHT_JSON = os.path.join(R2, 'preflight_result.json')
inst.a.file_loc = R2
inst.a.set_folder()

# --- widen the InvOLS sanity window (see docstring) --------------------------
old_bounds = inst.a.invols_bounds
inst.a.invols_bounds = (1e-8, 5e-5)

# --- seed a prior V_cpd so the chain is never blocked on a missing file ------
prior = dict(v_cpd_V=R1_CLOSEOUT_VCPD, source='R1_closeout_prior',
             note='seeded by 00_setup_r2.py; overwritten by 11_ and 55_',
             written=time.strftime('%Y-%m-%d %H:%M:%S'))
with open(os.path.join(R2, 'vcpd_current.json'), 'w', encoding='utf-8') as f:
    json.dump(prior, f, indent=1)

LOG = os.path.join(R2, 'setup_r2_log.txt')
_tee = C.Tee(LOG); _old = sys.stdout; sys.stdout = _tee
try:
    C.stamp('=' * 78)
    C.stamp('R2 SETUP -- repeat campaign on the SAME SCM-PIT probe, no load ladder')
    C.stamp('=' * 78)
    C.stamp(f'data folder : {OLD_FILE_LOC}')
    C.stamp(f'           -> {C.FILE_LOC}')
    C.stamp(f'preflight   -> {C.PREFLIGHT_JSON}')
    C.stamp(f'igor saves  -> {inst.a.file_loc}  (set_folder applied)')
    C.stamp(f'base name   : {inst.a.base_filename}  (file numbering restarts in the new folder)')
    C.stamp(f'invols_bounds {old_bounds} -> {inst.a.invols_bounds}  '
            f'(R1 close-out measured 1.993e-5 at the base and could not store it)')
    C.stamp(f'seeded vcpd_current.json with the R1 close-out prior {R1_CLOSEOUT_VCPD:+.3f} V')
    C.stamp('')
    C.stamp(f'instrument  : x={inst.current_x} um, spot={inst.current_spot}, '
            f'load={inst.load_nN} nN, bias={inst.dc_bias_V} V')
    C.stamp(f'x_limits_um : {inst.x_limits_um}  (R2 pre-flight will re-establish this)')
    C.stamp('')
    C.stamp('R1 reference values to beat / compare against:')
    C.stamp('   pre-flight CR1 285.50 kHz  ->  close-out 295.43 kHz (+3.56 %, uniform)')
    C.stamp('   if R2 pre-flight lands near 295 kHz the post-ladder tip state survived the withdraw;')
    C.stamp('   if it lands near 285 kHz the contact reset and R2 is a different tip state entirely.')
    C.stamp('   Either answer is useful -- but it changes how R2 should be read, so check the')
    C.stamp('   pre-flight number before trusting any cross-campaign comparison.')
    C.stamp('=' * 78)
    if not C.FILE_LOC.rstrip('\\/').endswith('_R2'):
        raise SystemExit(f'setup failed: C.FILE_LOC is {C.FILE_LOC!r}')
    C.stamp('R2 setup OK -- the queue may proceed')
finally:
    sys.stdout = _old; _tee.close()
