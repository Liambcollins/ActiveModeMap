"""Give every experiment folder a code/ subfolder and a MANIFEST.md."""
import collections, datetime, os, shutil

HOME = os.path.expanduser('~')
REPO = os.path.join(HOME, 'mnt/Liam/ActiveModeMap/scripts/scmpit')
R1 = os.path.join(HOME, 'mnt/ActiveModeMap/DomainsB_SCMPIT')
R2 = os.path.join(HOME, 'mnt/ActiveModeMap/DomainsB_SCMPIT_R2')

SHARED = ['scmpit_common.py']          # every stage imports this

# (stage, description, [acquisition scripts], [analysis scripts])
MAP_R1 = [
 ('00_probe_baseline', 'Free-air thermal calibration and pre-campaign images (before any contact)',
  [], []),
 ('01_preflight', 'Pre-flight: reachable span, CR1/CR2/CR3 at 500 nN, 19-position coarse walk',
  ['00_setup_inst.py', '01_preflight.py', 'relay.py'], []),
 ('10_bias_survey', 'Stage 1: two-domain bias survey, 8 pos x 7 biases x 2 domains, 500 nN, 1 V',
  ['10_bias_survey_500nN.py', 'analyze_stage1.py'],
  ['plot_stage1.py', 'fit_eb.py', 'fit_eb_joint.py', 'fit_eb_noE.py',
   'forward_model.py', 'higher_modes.py', 'plot_higher.py', 'plot_image.py']),
 ('20_wideband_fast', 'Stage 2a: live wideband capture, N=8 equispaced, 0 V, 500 nN',
  ['20_wideband_fast.py'], ['r2_wideband.py']),
 ('21_wideband_validation', 'Stage 2b: held-out validation at 7 never-visited midpoints',
  ['21_wideband_validation.py'], ['r2_wideband.py']),
 ('30_dense_grid', 'Stage 3: dense reference map, 181 positions at 1 um, 0 V, 500 nN',
  ['30_dense_grid_500nN.py'], ['r2an/common.py', 'r2an/s1_structure.py']),
 ('40_load_ladder', 'Stage 4: load ladder, 8 pos x 7 loads (50-1500 nN), spot 1, 0 V',
  ['40_load_ladder_spot1.py'], ['load_dependence.py', 'plot_load_dependence.py']),
 ('50_quant_imaging', 'Stage 5: quantitative CR1 imaging, positions A and B, 0 V and +0.77 V',
  ['50_image_cr1_quant.py'],
  ['analyze_stage5.py', 'analyze_stage5_v2.py', 'plot_stage5.py', 'plot_resonance_images.py',
   'plot_resonance_images_quant.py', 'plot_noise_comparison.py', 'r2_images.py', 'r2_maps.py']),
 ('60_lowac_30mV', 'Stage 6: low-drive imaging at Vac = 30 mV, position A',
  ['60_lowac_cr1_A.py'], ['analyze_stage6.py', 'plot_stage6.py', 'r2_images.py']),
 ('70_ac_series', 'Stage 7: 15-point AC drive series, 5 mV - 2 V, position A, both domains',
  ['70_ac_series_A.py'],
  ['analyze_stage7.py', 'analyze_stage7_v2.py', 'plot_stage7.py',
   'r2an/common.py', 'r2an/s2_ac_closeout.py']),
 ('80_closeout', 'Stage 8: close-out calibration - stage-1 repeat, mode-shape walk, final InvOLS',
  ['80_closeout_cal.py'],
  ['analyze_stage8.py', 'plot_stage8.py', 'r2an/common.py', 'r2an/s2_ac_closeout.py']),
]

MAP_R2 = [
 ('00_setup', 'R2 setup: repoint the campaign at this folder, widen the InvOLS bound, seed the V_cpd prior',
  ['r2/00_setup_r2.py', 'relay.py'], []),
 ('01_preflight', 'Pre-flight: reachable span, CR1/CR2/CR3 at 500 nN, 17-position coarse walk',
  ['r2/01_preflight.py'], []),
 ('10_bias_survey', 'Stage 1: two-domain bias survey, 8 pos x 7 biases x 2 domains, 500 nN, 1 V',
  ['r2/10_bias_survey_500nN.py'], ['r2an/common.py', 'r2an/s1_structure.py']),
 ('11_vcpd_from_stage1', "Stage 1b: V_cpd recovered from R2's own bias survey (analysis only)",
  ['r2/11_vcpd_from_stage1.py'], []),
 ('20_wideband_fast', 'Stage 2a: live wideband capture, N=8 equispaced, 0 V, 500 nN',
  ['r2/20_wideband_fast.py'], ['r2_wideband.py']),
 ('21_wideband_validation', 'Stage 2b: held-out validation at 7 never-visited midpoints',
  ['r2/21_wideband_validation.py'], ['r2_wideband.py']),
 ('50_quant_imaging', 'Stage 5: quantitative CR1 imaging, positions A and B, three bias points at A',
  ['r2/50_image_cr1_quant_r2.py'], ['r2_images.py', 'r2_maps.py']),
 ('55_vcpd_from_images', 'Stage 5b: V_cpd refined from the stage-5 image series (analysis only)',
  ['r2/55_vcpd_from_images.py'], []),
 ('60_lowac_30mV', 'Stage 6: low-drive imaging at Vac = 30 mV at the measured null',
  ['r2/60_lowac_cr1_A_r2.py'], ['r2_images.py']),
 ('70_ac_series', 'Stage 7: 17-point AC drive series, 2 mV - 2 V, position A, both domains',
  ['r2/70_ac_series_A_r2.py'],
  ['r2an/common.py', 'r2an/s2_ac_closeout.py', 'r2_compare.py']),
 ('75_dense_grid', 'Stage 3 (run last in R2): dense reference map, 161 positions at 1 um',
  ['r2/75_dense_grid_500nN.py'], ['r2an/common.py', 'r2an/s1_structure.py']),
 ('80_closeout', 'Stage 8: close-out calibration - stage-1 repeat, mode-shape walk, final InvOLS',
  ['r2/80_closeout_cal_r2.py'],
  ['r2an/common.py', 'r2an/s2_ac_closeout.py', 'analyze_stage8.py', 'plot_stage8.py']),
 ('40_load_ladder', 'R2 extension stage 4: load ladder, 5 loads (50-500 nN) x 8 positions, 0 V, '
  'with a full probe check after every load',
  ['r2/40_load_ladder_r2.py'],
  ['r2an/common.py', 'r2an/s4_load.py', 'load_stiffness_r2.py', 'forward_model.py',
   'paper_fig2.py']),
 ('41_bias_vs_load', 'R2 extension stage 4b: two-domain bias series (7 biases) at 100, 300 and '
  '500 nN, working position -- V_cpd, |b|/|P| and d33 versus load',
  ['r2/41_bias_vs_load_r2.py'],
  ['r2an/common.py', 'r2an/s4_load.py', 'paper_fig2.py']),
 ('42_postcheck', 'R2 extension stage 4c: 8-position mode-shape walk at 0 V / 500 nN, to prove '
  'the load excursion left the contact state unchanged',
  ['r2/42_postcheck_r2.py'],
  ['r2an/common.py', 'r2an/s4_load.py']),
]


def src_path(name, kind):
    base = REPO if kind == 'acq' else os.path.join(REPO, 'analysis')
    return os.path.join(base, name)


def copy_in(dst_dir, name, kind):
    src = src_path(name, kind)
    if not os.path.exists(src):
        return None, f'MISSING {src}'
    dst = os.path.join(dst_dir, os.path.basename(name))
    if os.path.exists(dst) and os.path.getsize(dst) == os.path.getsize(src):
        return os.path.basename(name), None
    try:
        shutil.copyfile(src, dst)
    except OSError as e:
        return None, f'{name}: {e}'
    return os.path.basename(name), None


def do(root, mapping, campaign):
    print(f'\n==== {campaign} ====')
    problems = []
    for stage, desc, acq, ana in mapping:
        d = os.path.join(root, stage)
        if not os.path.isdir(d):
            print(f'  {stage:24s} MISSING FOLDER'); continue
        code = os.path.join(d, 'code')
        os.makedirs(code, exist_ok=True)
        got_acq, got_ana = [], []
        for n in acq + (SHARED if acq else []):
            nm, err = copy_in(code, n, 'acq')
            (got_acq.append(nm) if nm else problems.append(err))
        for n in ana:
            nm, err = copy_in(code, n, 'ana')
            (got_ana.append(nm) if nm else problems.append(err))

        data = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))
                      and f != 'MANIFEST.md')
        tms = [os.path.getmtime(os.path.join(d, f)) for f in data] or [os.path.getmtime(d)]
        t0 = datetime.datetime.fromtimestamp(min(tms)) - datetime.timedelta(hours=4)
        t1 = datetime.datetime.fromtimestamp(max(tms)) - datetime.timedelta(hours=4)
        ext = collections.Counter(os.path.splitext(f)[1] for f in data)
        size = sum(os.path.getsize(os.path.join(d, f)) for f in data)
        key = [f for f in data if os.path.splitext(f)[1] in ('.npz', '.json', '.npy')
               or f.endswith('_log.txt')]

        L = [f'# {campaign} · {stage}', '', desc, '',
             '- **data files:** %d  (%s)' % (len(data), ', '.join(
                 '%d %s' % (v, k or '(no ext)') for k, v in sorted(ext.items()))),
             f'- **size:** {size / 1e6:.1f} MB',
             f'- **acquired:** {t0:%Y-%m-%d %H:%M} – {t1:%H:%M} instrument local', '']
        if key:
            L += ['## Results, logs and checkpoints', ''] + [f'- `{f}`' for f in key] + ['']
        L += ['## code/', '']
        if not got_acq and not got_ana:
            L += ['No script produced this folder: it predates the automated campaign '
                  '(free-air thermal calibration and pre-contact images, taken by hand at the '
                  'instrument). Nothing is missing.', '']
        if got_acq:
            L += ['Acquisition — what produced this data, exactly as it ran:', '']
            L += [f'- `{f}`' for f in got_acq] + ['']
        if got_ana:
            L += ['Analysis — what was used to turn it into results:', '']
            L += [f'- `{f}`' for f in got_ana] + ['']
        L += ['Scripts here are copies; the working versions live in',
              '`…\\Liam\\ActiveModeMap\\scripts\\scmpit\\` (acquisition, `r2\\` for the R2 variants)',
              'and `…\\scripts\\scmpit\\analysis\\` (analysis). Acquisition scripts run inside the',
              'Igor kernel through the file relay and need `scmpit_common.py` plus the',
              '`activemodemap` package; analysis scripts are plain numpy/matplotlib and also need',
              'igor2 (image frames), and scipy + scikit-learn for the wideband and imaging ones.', '',
              '## File types', '',
              '- `Tune_SCMPIT_X*.txt` — one saved tune (full complex spectrum) per position/condition',
              '- `FSCMPIT*.ibw` — force curve behind each InvOLS measurement',
              '- `FSCMPIT*.tif` — optical image confirming where the laser landed',
              '- `*_checkpoint.npz` — the stage\'s complex spectra, resumable',
              '- `*_log.txt` — the run log, the authoritative timeline for this stage', '']
        open(os.path.join(d, 'MANIFEST.md'), 'w', encoding='utf-8').write('\n'.join(L))
        print(f'  {stage:24s} data {len(data):4d}  code {len(got_acq) + len(got_ana):2d}')
    for p in problems:
        print('  PROBLEM:', p)
    return problems


p1 = do(R1, MAP_R1, 'Run 1 (2026-09-20)')
p2 = do(R2, MAP_R2, 'Run 2 (2026-09-21)')
print(f'\nproblems: {len(p1) + len(p2)}')
