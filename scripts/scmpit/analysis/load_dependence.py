import numpy as np, json

LOADS = [50, 150, 300, 500, 750, 1000, 1500]
BASE = '/mnt/user-data/uploads/ActiveModeMap/DomainsB_SCMPIT/load_ladder_spot1_{:04d}nN_checkpoint.npz'
CR1_WIN = (230e3, 340e3)   # preflight CR1 band, wide enough to track the +3.3% stiffening at 1500 nN

xt = None
freq = None
amp = np.zeros((len(LOADS), 8))     # |Z| at CR1 peak
fpk  = np.zeros((len(LOADS), 8))    # CR1 peak frequency
ph   = np.zeros((len(LOADS), 8))    # phase at peak (deg)
qfac = np.zeros((len(LOADS), 8))    # crude Q from FWHM

def q_from(fr, a):
    j = int(np.argmax(a)); h = a[j]/np.sqrt(2); lo = j
    while lo > 0 and a[lo] > h: lo -= 1
    hi = j
    while hi < a.size-1 and a[hi] > h: hi += 1
    return fr[j] / max(fr[hi]-fr[lo], 1)

for i, L in enumerate(LOADS):
    d = np.load(BASE.format(L), allow_pickle=True)
    if xt is None:
        xt = d['x_um']; freq = d['freq_Hz']
        order = np.argsort(xt)          # ladder logs free-end-first; sort ascending for plotting
    Z = d['Z'][0]                        # (8 positions, nfreq) complex, condition 0 = the only one
    m = (freq >= CR1_WIN[0]) & (freq <= CR1_WIN[1])
    fr = freq[m]
    for j in range(8):
        a = np.abs(Z[j, m])
        k = int(np.argmax(a))
        amp[i, j] = a[k]
        fpk[i, j] = fr[k]
        ph[i, j] = np.degrees(np.angle(Z[j, m][k]))
        qfac[i, j] = q_from(fr, a)

xt = xt[order]; amp = amp[:, order]; fpk = fpk[:, order]; ph = ph[:, order]; qfac = qfac[:, order]
np.savez('load_dependence.npz', loads=np.array(LOADS), xt=xt, amp=amp, fpk=fpk, ph=ph, qfac=qfac)
print('xt (base->tip):', xt)
print('\namp (mV) [load x position]:')
print(np.array2string(amp*1e3, precision=3, suppress_small=True))
print('\nfpk (kHz):')
print(np.array2string(fpk/1e3, precision=2))
print('\nQ (crude, FWHM in this narrow window -- not reliable where the band clips the line):')
print(np.array2string(qfac, precision=0))
