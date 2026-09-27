r"""Build the blind-EB-fit inputs for Run 2 from s3_transfer.json + the pre-flight tune.

Only quantities that the *blind* fit is allowed to see go into the residual:
  - the three contact-resonance frequencies from the pre-flight wideband tune,
  - the static shape 1/InvOLS(x) (shape only; the overall gain is projected out),
  - the CR1 quality factor.
The measured enhancement E(x) and every amplitude are carried along for scoring
only, exactly as in the companion (Run 1) analysis.
"""
import json

import numpy as np

UP = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT_R2'
TR = json.load(open(f'{UP}/analysis/s3_transfer.json'))
S1 = json.load(open(f'{UP}/s1_structure.json'))

rows = TR['rows']
X = np.array([r['x_um'] for r in rows], float)
o = np.argsort(X)
rows = [rows[i] for i in o]
X = X[o]
inv = np.array([r['invols'] for r in rows])
Q_x = np.array([r['Q'] for r in rows])
f_cr1_x = np.array([r['f_cr1_Hz'] for r in rows])
P_qs = np.array([r['P_qs'] for r in rows])
P_cr1 = np.array([r['P_cr1'] for r in rows])
E_meas = np.array([r['E'] for r in rows])

pf = S1['preflight']['R2']
CR = np.array([pf['cr1'], pf['cr2'], pf['cr3']], float)

# The Run-2 static fit hit its physical lower bound L >= max(x) - x0, so L is a
# bound, not a determination.  Refit x0 at the Run-1 lever length as a sensitivity
# case; both are handed to the fitter.
y = 1.0 / inv


def static_fit(L_fixed=None):
    best = None
    for x0 in np.arange(-40, 60, 0.05):
        s = X - x0
        if s.min() <= 0:
            continue
        Ls = [L_fixed] if L_fixed else np.arange(max(190.0, s.max()), 300, 0.5)
        for L in Ls:
            if L < s.max():
                continue
            m = s ** 2 * (3 * L - s)
            g = np.sum(y * m) / np.sum(m * m)
            rms = float(np.sqrt(np.mean((y / (g * m) - 1) ** 2)))
            if best is None or rms < best[0]:
                best = (rms, x0, L)
    return best


rms_free, x0_free, L_free = static_fit()
rms_225, x0_225, _ = static_fit(225.89307802735)   # Run-1 lever length
print(f'static shape, free L : x0 {x0_free:+.2f} um, L {L_free:.1f} um, rms {100*rms_free:.2f} %'
      f'   (L at lower bound: {L_free <= X.max()-x0_free+1e-6})')
print(f'static shape, L=225.9: x0 {x0_225:+.2f} um, rms {100*rms_225:.2f} %')

np.savez('r2_eb_inputs.npz',
         X=X, invols=inv, static_meas=y,
         cr1=CR[0], cr2=CR[1], cr3=CR[2], CR=CR,
         f0_free=63801.0, k_lever=1.697, vac=1.0,
         Q_x=Q_x, f_cr1_x=f_cr1_x, P_qs=P_qs, P_cr1=P_cr1, E_meas=E_meas,
         x0_pos=x0_free, L_true=L_free, calib_rms=rms_free,
         x0_alt=x0_225, L_alt=225.89307802735, calib_rms_alt=rms_225)
print('wrote r2_eb_inputs.npz')
print(f'  Q median (excl. free end) {np.median(Q_x[:-1]):.0f}; E span '
      f'{E_meas.min():.1f}-{E_meas.max():.1f}')
