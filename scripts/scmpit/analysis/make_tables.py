r"""Generate the manuscript's tables from the reduced JSON, so no number is transcribed by hand."""
import json

import numpy as np

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT_R2'
TR = json.load(open(f'{UP}/analysis/s3_transfer.json'))
S4 = json.load(open(f'{UP}/analysis/s4_load.json'))
WB = json.load(open('/home/claude/scmpit_analysis/r2_wideband.json'))
IM = json.load(open('/home/claude/scmpit_analysis/r2_images.json'))
NF = json.load(open('/home/claude/scmpit_analysis/r2_noisefloor.json'))
LS = json.load(open('/home/claude/scmpit_analysis/r2_load_stiffness.json'))

out = ['## Tables', '']

# --- Table 1: stages -------------------------------------------------------
stages = [
    ('Pre-flight', '17 positions, 10 µm steps, InvOLS + tune at each', '20.8'),
    ('Bias survey', '8 positions × 7 biases × 2 domains, 500 nN, 1 V', '45.5'),
    ('Wideband capture', '8 equispaced positions, 0 V, 500 nN', '9.6'),
    ('Wideband validation', '7 held-out midpoints', '8.2'),
    ('Quantitative imaging', '9 frames, 2 positions, 3 biases', '42.4'),
    ('Low-drive imaging', 'position A repeated at $V_{ac}$ = 30 mV', '14.7'),
    ('Drive series', '17 amplitudes, 2 mV – 2 V, both domains', '42.1'),
    ('Dense reference map', '161 positions at 1 µm', '192.4'),
    ('Close-out', 'reference condition, walk, final InvOLS', '16.2'),
    ('*Load ladder*', '5 loads × 8 positions, probe check after each', '52.7'),
    ('*Bias × load*', '3 loads × 7 biases × 2 domains, one position', '16.8'),
    ('*Post-load walk*', '8 positions, 0 V, 500 nN', '9.5'),
]
tot = sum(float(c) for _, _, c in stages[:9])
tot_ext = sum(float(c) for _, _, c in stages[9:])
out += [f'**Table 1.** Measurement stages, wall clock. The overnight campaign (first nine) ran '
        f'22:56-05:28 for {tot:.0f} min with zero failures; the last three (italic) are the load '
        f'extension of §3.2, {tot_ext:.0f} min, run the following morning. Two analysis-only '
        f'stages that derive $V_{{cpd}}$ from the preceding data are not listed.', '',
        '| stage | design | wall clock (min) |', '|---|---|---|']
out += [f'| {a} | {b} | {c} |' for a, b, c in stages]
out += ['']

# --- Table 2: the transfer function ---------------------------------------
out += ['**Table 2.** Cantilever transfer function from the Run 2 bias survey: eight laser '
        'positions, 500 nN, $V_{ac}$ = 1 V, $\\pm 9$ V in seven steps on both domains. '
        '$|P|$ is the decomposed piezoresponse in detector volts; $E = |P|_{CR1}/|P|_{qs}$.', '',
        '| $x$ from clamp (µm) | InvOLS (m/V) | CR1 (kHz) | $Q$ | $|P|_{qs}$ (V) | '
        '$|P|_{CR1}$ (V) | $E$ | $V_{cpd}$ (V) | $|b|/|P|$ | $d_{33}$ (pm/V) |',
        '|---|---|---|---|---|---|---|---|---|---|']
for r in sorted(TR['rows'], key=lambda r: r['x_clamp']):
    out.append(f'| {r["x_clamp"]:.1f} | {r["invols"]:.3e} | {r["f_cr1_Hz"]/1e3:.2f} | '
               f'{r["Q"]:.0f} | {r["P_qs"]:.3e} | {r["P_cr1"]:.3e} | {r["E"]:.1f} | '
               f'{r["vcpd_qs"]:+.3f} | {r["ratio_qs"]:.3f} | {r["d33_qs_pm_per_V"]:.2f} |')
d = np.array([r['d33_qs_pm_per_V'] for r in TR['rows']])
out += ['', f'Median $d_{{33}}$ {np.median(d):.2f} pm/V over a {max(r["E"] for r in TR["rows"]) / min(r["E"] for r in TR["rows"]):.0f}× '
        f'range of enhancement; position calibration $x_0$ = {TR["x0_um"]:+.1f} µm, '
        f'{100*TR["calib_rms"]:.1f} % rms.', '']

# --- Table 3: wideband -----------------------------------------------------
ranks = sorted(int(k) for k in WB['R2']['scores'])
out += ['**Table 3.** Low-rank wideband reconstruction, scored on seven never-visited positions '
        'over 100 kHz – 1.9 MHz. Both campaigns through identical code.', '',
        '| rank | R2 NRMSE | R2 within 3 dB | R2 phase MAE | R2 CR3 band | R1 NRMSE | R1 CR3 band |',
        '|---|---|---|---|---|---|---|']
for k in ranks:
    a, b = WB['R2']['scores'][str(k)], WB['R1']['scores'][str(k)]
    out.append(f'| {k} | {a["nrmse"]:.3f} | {100*a["frac_within_3db"]:.0f} % | '
               f'{a["phase_mae"]:.1f}° | {a["per_mode"]["cr3"]:.3f} | '
               f'{b["nrmse"]:.3f} | {b["per_mode"]["cr3"]:.3f} |')
out += ['']

# --- Table 4: imaging ------------------------------------------------------
fr = IM['R2']['frames']
order = [('A', 'quasi-static 20 kHz', 'A_qs20k_0V'), ('A', 'CR1', 'A_cr1_0V'),
         ('A', 'CR1', 'A_cr1_Vhalf'), ('A', 'CR1 (null)', 'A_cr1_Vcpd'),
         ('B', 'quasi-static 20 kHz', 'B_qs20k_0V'), ('B', 'CR1', 'B_cr1_0V'),
         ('B', 'CR1 (null)', 'B_cr1_Vcpd')]
out += ['**Table 4.** Quantitative imaging, Run 2. Every $d_{33}$ is amplitude ÷ ($V_{ac}$ × the '
        'in-situ enhancement measured for that domain); quasi-static frames need only ÷ $V_{ac}$.',
        '', '| position | channel | bias (V) | $d_{33}$ domain 1 | $d_{33}$ domain 2 | '
        'amplitude ratio | phase separation |', '|---|---|---|---|---|---|---|']
for pos, chan, key in order:
    if key not in fr:
        continue
    f = fr[key]
    out.append(f'| {pos} | {chan} | {f["bias"]:+.3f} | {f.get("d33_1", float("nan")):.2f} | '
               f'{f.get("d33_2", float("nan")):.2f} | {f["ratio"]:.3f} | '
               f'{f["sep"]:.1f}° |')
out += ['']

# --- Table 5: noise floor --------------------------------------------------
out += ['**Table 5.** Magnitude-detection floor, $|Z| = \\sqrt{(kV_{ac})^2 + n_0^2}$, fitted to '
        'the 17-point drive series over three decades.', '',
        '| series | $k$ (mV/V) | $n_0$ (µV) | signal = floor at | median residual |',
        '|---|---|---|---|---|']
for tag, lab in (('R2_s1', 'Run 2, domain 1'), ('R2_s2', 'Run 2, domain 2'),
                 ('R1_s1', 'companion, domain 1'), ('R1_s2', 'companion, domain 2')):
    e = NF[tag]
    out.append(f'| {lab} | {e["k"]*1e3:.1f} | {e["n0"]*1e6:.0f} | '
               f'{1e3*e["n0"]/e["k"]:.1f} mV | {100*e["resid"]:.1f} % |')
out += ['']

# --- Table 6: the load ladder ---------------------------------------------
out += ['**Table 6.** Load ladder. $E$ is the median over the positions that returned a '
        'resonance; $k^*$ is inverted from CR1 with the blind-fit geometry of §3.4 held fixed, '
        'and is undetermined once the measurement reaches the model\'s stiff-contact asymptote '
        f'({LS["model_asymptote_Hz"]/1e3:.1f} kHz).', '',
        '| load (nN) | positions with CR1 | CR1 (kHz) | $Q$ | median $E$ | '
        '$k^*/k_{\\mathrm{lever}}$ | $\\mathrm{d}\\ln f/\\mathrm{d}\\ln k^*$ |',
        '|---|---|---|---|---|---|---|']
for r in LS['rows']:
    k = f'{r["k_ratio"]:.0f}' if r['determined'] else 'saturated'
    sens = f'{r["dlnf_dlnk"]:.4f}' if r['determined'] else '< 0.001'
    out.append(f'| {r["load_nN"]:.0f} | {r["n_pos"]}/8 | {r["f_cr1_Hz"]/1e3:.2f} ± '
               f'{r["f_sd_Hz"]:.0f} Hz | {r["Q"]:.0f} | {r["E_med"]:.0f} | {k} | {sens} |')
out += ['']
bvl = S4['bias_vs_load']
out += ['**Table 7.** Bias series at three loads, working position (154.9 stage-µm), '
        '$\\pm 9$ V in seven steps on both domains. The quasi-static decomposition passes the '
        'checks of §2.5 at every load; the CR1 decomposition fails all three (flip angle far '
        'from 180°, $|b_2/b_1|$ far from 1), so no CR1 quantity from this stage is quoted. '
        'The last column applies the InvOLS the probe checks measured, in place of the '
        'pre-flight curve.', '',
        '| load (nN) | $V_{cpd}$ quasi-static (V) | $|P|_{qs}$ (V) | $|b|/|P|$ | flip angle | '
        '$|b_2/b_1|$ | CR1 fit | $d_{33}$ (pm/V) | $d_{33}$, measured InvOLS |',
        '|---|---|---|---|---|---|---|---|---|']
for k in sorted(bvl, key=float):
    r = bvl[k]
    crf = (f'fails: {r["cr1"]["flip_deg"]:.0f}°, {r["cr1"]["b_bal"]:.2f}'
           if not r['cr1'].get('checks_pass') else 'passes')
    out.append(f'| {float(k):.0f} | {r["qs"]["v_cpd"]:+.3f} | {r["qs"]["P_abs"]:.3e} | '
               f'{r["qs"]["ratio"]:.3f} | {r["qs"]["flip_deg"]:.1f}° | '
               f'{r["qs"]["b_bal"]:.3f} | {crf} | {r["d33_qs_pm_per_V"]:.2f} | '
               f'{r["d33_qs_drift_corrected"]:.2f} |')
dr = S4['invols_drift']
out += ['', '**Table 8.** InvOLS re-measured at the reference position (232 stage-µm) after each '
        'load step, against the pre-flight curve used throughout the reduction. The pre-flight '
        'was taken nine hours earlier.', '',
        '| after load (nN) | measured InvOLS (m/V) | pre-flight (m/V) | ratio |',
        '|---|---|---|---|']
for d in dr['rows']:
    out.append(f'| {d["after_load_nN"]:.0f} | {d["measured"]:.4e} | {d["preflight"]:.4e} | '
               f'{d["ratio"]:.3f} |')
out += ['', f'Mean {dr["mean_ratio"]:.3f}, drifting {dr["span_pct"]:.1f} % across the series.', '']

open('/home/claude/scmpit_analysis/paper_tables.md', 'w').write('\n'.join(out))
print('wrote paper_tables.md,', len(out), 'lines')
