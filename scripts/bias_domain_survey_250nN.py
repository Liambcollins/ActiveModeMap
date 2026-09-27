# =====================================================================================
# Two-domain bias survey at 250 nN -- queued 2026-09-19, to run after bias_image_series.py
#
# WHY.  The 15 nN TipBias survey (domains_bias_checkpoint.npz) found V_cpd = +1.51 V,
# mode-independent, and a channel ratio |b|/|P| (electrostatic per volt of V-V_cpd,
# over piezoresponse) that falls steeply with mode number: CR1 1.64, CR2 0.24,
# CR3 0.12, CR4 0.059. The electrostatic force loads against the CONTACT stiffness
# (b ~ dC/dz / k_eff), while the piezoresponse P is ~load-independent to first order.
# Raising the load raises the Hertzian contact stiffness, so |b|/|P| should fall at
# every mode if the mixture-model mechanism (pppcontau-electrostatic-vs-contact-drive)
# is right; a background/artifact term would not track load this way.
# 250 / 15 nN = 16.7x the force -> ~16.7^(1/3) = 2.55x the Hertzian contact stiffness,
# if contact stiffness (not lever stiffness) sets k_eff -- the scale this run tests.
#
# EXACT SAME DESIGN as the 15 nN survey (notebook 07, cell 4 + cells 26/30) -- only
# load_nN and the checkpoint path change, so the two runs are directly comparable:
#   x        = 8 positions, 75 -> 445 um (SPARSE_X_UM, identical values)
#   bias     = [-9,-6,-3,0,3,6,9] V, interleaved (drift -> scatter, not slope)
#   spots    = domain 1, domain 2 (outermost loop -- one spot hop, not sixteen)
#   14 conditions/position x 8 positions, ~40 min total (same estimator as cell 4)
#
# Deflection at 250 nN with k = 0.14818 N/m: 1.69 um, vs 101 nm at 15 nN -- a real
# force increase, well past the 200 nN top of the notebook's own ascending load
# series (cell 14) but consistent with the up-to-450-nN range that series' own
# comment anticipated. set_load(reengage=True) lifts and re-engages for the jump,
# which is asylum.py's own rule for a large setpoint step. DNS_BAND_HZ is widened
# past the load-series band (55-85 kHz) since CR1 stiffens toward the pinned limit
# with load and 250 nN exceeds every load tested there.
#
# This script only ACQUIRES and does a quick frequency sanity check. The full
# per-mode V_cpd / |b|/|P| analysis (same method as tipbias-survey-analysis-2026-09-18:
# per-spot complex line fit in V, combined as V_cpd=-(a1+a2)/2b, P=(a1-a2)/2,
# b=(b1+b2)/2) is a separate, careful pass -- done by the scheduled check-in, not here.
# =====================================================================================
import os, sys, time, contextlib
import numpy as np
from activemodemap import (make_conditions, run_series, separate_domains,
                            separate_channels, channel_spots, estimate_v_cpd,
                            load_checkpoint)

file_loc = r'D:\User Data\Liam\ActiveModeMap\DomainsBPPPCONTAU'
os.makedirs(file_loc, exist_ok=True)
LOG = os.path.join(file_loc, 'bias_domain_survey_250nN_log.txt')


class _Tee:
    """Mirror stdout to the log file too, so run_series'/measure_conditions_at's own
    bare print()s land in the log exactly like the dense-map/sparse-capture logs do,
    not just the say() lines below."""
    def __init__(self, *streams):
        self.streams = streams

    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()

    def flush(self):
        for st in self.streams:
            st.flush()


_lf = open(LOG, 'a', buffering=1)
_tee = _Tee(sys.__stdout__, _lf)


def say(m=''):
    print(f'{time.strftime("%H:%M:%S")}  {m}', flush=True)


# ---- exact match to the 15 nN survey's design (notebook 07, cell 4) -----------------
PROBE_L_UM      = 445.0
X_LO_UM         = 75.0
K_LEVER_N_PER_M = 0.14818          # recalibrated 2026-09-18 thermal value
BIAS_V          = [-9, -6, -3, 0, 3, 6, 9]
BIAS_ORDER      = 'interleaved'
SPOT_UP, SPOT_DOWN = 1, 2
REF_INDEX       = 0
RANK            = 4
DNS_BAND_HZ     = (55e3, 100e3)    # widened: 250 nN exceeds every load in the 55-85 kHz band
SPARSE_X_UM     = np.round(np.linspace(X_LO_UM, PROBE_L_UM, 8), 1)   # identical to the 15 nN run
SPARSE_ORDER    = SPARSE_X_UM[::-1]                                   # free end first

# ---- the one thing that changes -----------------------------------------------------
LOAD_NN         = [250.0]          # true newtons at k = 0.14818 N/m; was [14.818] (~15 nN)
CHECKPOINT      = os.path.join(file_loc, 'domains_bias_checkpoint_250nN.npz')
REF_CHECKPOINT  = os.path.join(file_loc, 'domains_bias_checkpoint.npz')   # the 15 nN run

deflect_250_um = 250.0e-9 / K_LEVER_N_PER_M * 1e6
deflect_15_um  = 14.818e-9 / K_LEVER_N_PER_M * 1e6

say('=' * 78)
say(f'TWO-DOMAIN BIAS SURVEY AT 250 nN  (queued {time.strftime("%Y-%m-%d")}; '
    f'runs after bias_image_series.py)')
say(f'deflection: {deflect_250_um:.2f} um at 250 nN vs {deflect_15_um:.2f} um at the '
    f'15 nN reference run (k = {K_LEVER_N_PER_M} N/m)')
say('=' * 78)

conditions = make_conditions(BIAS_V, LOAD_NN, spots=[SPOT_UP, SPOT_DOWN],
                              bias_order=BIAS_ORDER)
say(f'{len(conditions)} conditions/position: ' + ', '.join(str(c) for c in conditions))
say(f'positions: {SPARSE_X_UM.tolist()}')
say(f'measurement order (free end first): {SPARSE_ORDER.tolist()}')

try:
    with contextlib.redirect_stdout(_tee):
        series = run_series(inst, SPARSE_X_UM, conditions, ref_index=REF_INDEX,
                             rank=RANK, dns_band_Hz=DNS_BAND_HZ,
                             checkpoint_path=CHECKPOINT, resume=True,
                             positions_um=SPARSE_ORDER, verbose=True)
    say(f"\n{series['x_um'].size} positions x {len(conditions)} conditions x "
        f"{series['freq_Hz'].size} frequencies -- survey done")

    # ---- quick sanity check: did CR1 stiffen the way higher load should? -----------
    if os.path.exists(REF_CHECKPOINT):
        ref = load_checkpoint(REF_CHECKPOINT)
        F, Zref, Znew = ref['freq_Hz'], ref['Z'], series['Z']
        band = (F > 55e3) & (F < 100e3)
        f_cr1_15nN = F[band][np.argmax(np.abs(Zref[REF_INDEX, :, band]).mean(axis=0))]
        f_cr1_250nN = F[band][np.argmax(np.abs(Znew[REF_INDEX, :, band]).mean(axis=0))]
        say(f'\nCR1 (0 V, spot {SPOT_UP}, mean over positions): '
            f'{f_cr1_15nN/1e3:.2f} kHz at 15 nN -> {f_cr1_250nN/1e3:.2f} kHz at 250 nN '
            f'({(f_cr1_250nN - f_cr1_15nN)/1e3:+.2f} kHz)')
        say('  (a stiffening shift upward is expected; a flat or falling frequency '
            'is worth a second look before trusting the |b|/|P| comparison)')
    else:
        say(f'\n15 nN reference checkpoint not found at {REF_CHECKPOINT} -- '
            'skipping the frequency sanity check')

except Exception as e:
    say(f'\n*** FAILED: {type(e).__name__}: {e}')
    import traceback
    traceback.print_exc()
    raise
finally:
    try:
        inst.set_dc_bias(0.0)
    except Exception as e:
        say(f'  (bias reset failed: {e})')
    try:
        inst.a.withdraw()
    except Exception as e:
        say(f'  (withdraw failed: {e})')
    say(f'tip withdrawn, bias 0 V. checkpoint -> {CHECKPOINT}')
    say('=' * 78)
    _lf.close()

bias_domain_survey_250nN_out = series
