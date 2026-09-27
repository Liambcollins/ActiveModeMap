import numpy as np, re
from igor2.binarywave import load as ibw_load

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT/'
FILES = {
    'A_qs20k_0V':      ('QImg0573.ibw', 154.9, 0.0,  20000.0, 1.0),
    'A_cr1_0V':        ('QImg0574.ibw', 154.9, 0.0,  294815.72, 1.0),
    'A_cr1_Vcpd':      ('QImg0575.ibw', 154.9, 0.77, 294815.72, 1.0),
    'A_cr1_0V_repeat': ('QImg0576.ibw', 154.9, 0.0,  294815.72, 1.0),
    'B_qs20k_0V':      ('QImg0578.ibw', 232.0, 0.0,  20000.0, 1.0),
    'B_cr1_0V':        ('QImg0579.ibw', 232.0, 0.0,  295155.78, 1.0),
    'B_cr1_Vcpd':      ('QImg0580.ibw', 232.0, 0.77, 295155.78, 1.0),
    'B_qs20k_0V_2Vac': ('QImg0581.ibw', 232.0, 0.0,  20000.0, 2.0),
}
E_BY_POS = {154.9: 200.2, 232.0: 2.8}

def parse_note(note):
    d = {}
    for line in note.decode(errors='ignore').split('\r'):
        if ':' in line:
            k, v = line.split(':', 1)
            try: d[k.strip()] = float(v.strip())
            except ValueError: pass
    return d

data = {}
for name, (fn, x, vdc, fdr, vac) in FILES.items():
    w = ibw_load(UP + fn)['wave']
    note = parse_note(w['note'])
    amp = w['wData'][:, :, 1].astype(float)     # AmplitudeRetrace, volts
    phase = w['wData'][:, :, 3].astype(float)   # PhaseRetrace, degrees
    invols = note.get('InvOLS', np.nan)
    data[name] = dict(amp=amp, phase=phase, invols=invols, x=x, vdc=vdc, fdr=fdr, vac=vac,
                       drive_amplitude_note=note.get('DriveAmplitude'), spring=note.get('SpringConstant'))
    print(f"{name:20s} InvOLS(note)={invols:.4e}  DriveAmplitude(note)={note.get('DriveAmplitude')}  "
          f"amp median={np.median(amp)*1e3:.3f} mV")

np.savez('stage5_raw.npz', **{f'{k}_amp': v['amp'] for k, v in data.items()},
         **{f'{k}_phase': v['phase'] for k, v in data.items()},
         meta=np.array([{k: {kk: vv for kk, vv in v.items() if kk not in ('amp','phase')} for k, v in data.items()}], dtype=object))
print('\nsaved stage5_raw.npz')
