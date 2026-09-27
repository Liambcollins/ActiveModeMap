"""Wide-band accuracy: where the held-out error lives across band A, by n.

Answers "can Physics+GP really reconstruct the wide-band response with n < 5?"
Note the structural fact: rec_eb_gp returns the bare-physics map for n < 5
(the GP needs 5 revealed points), so 'FEM+GP at n = 3-4' IS bare FEM.
"""
import numpy as np, pickle, matplotlib.pyplot as plt
import physrec as P, ornl as O
from run_phys import prep
O.style()
x,f,A=prep(); FK=f/1e3
out=pickle.load(open('out/wideband.pkl','rb'))
F=P.noise_floor(A); dB=lambda M:20*np.log10(M+F)

fig,(a1,a2)=plt.subplots(1,2,figsize=(13.2,4.35))
fig.subplots_adjust(left=.055,right=.985,top=.90,bottom=.135,wspace=.20)
series=[(('fem',3),'bare FEM, n = 3',O.ORANGE,'-',1.0),
        (('fem',4),'bare FEM, n = 4  (= "FEM+GP": GP is silent below n = 5)',O.ORANGE,'--',1.0),
        (('fem_gp',5),'FEM + GP, n = 5',O.GREEN,'-',.55),
        (('fem_gp',6),'FEM + GP, n = 6',O.GREEN,'-',.8),
        (('fem_gp',8),'FEM + GP, n = 8',O.GREEN,'-',1.0)]
for key,lab,c,ls,al in series:
    sel,Ar=out[key]
    held=np.array([i for i in range(len(x)) if i not in set(sel)])
    E=np.abs(dB(Ar[held])-dB(A[held])).mean(0)
    a1.plot(FK,E,ls,color=c,lw=1.9,alpha=al,label=lab)
a1.axhline(3,color=O.MUT,lw=1.0,ls=(0,(4,3)))
a1.text(.985,3.08,'3 dB ',color=O.INK2,fontsize=8.8,ha='right',va='bottom',
        transform=a1.get_yaxis_transform())
prof=A.max(0); w=prof>=.10*prof.max()
a1.axvspan(FK[w][0],FK[w][-1],color=O.GREEN,alpha=.08,lw=0)
a1.text(FK[w].mean(),4.7,'fit\nwindow',color='#004D21',fontsize=8.8,
        ha='center',va='bottom')
a1.set_xlabel('frequency  (kHz)')
a1.set_ylabel('mean |error| on held-out positions  (dB)')
a1.set_ylim(0,9.9)
a1.legend(fontsize=8.8,labelcolor=O.INK2,loc='upper left')
O.clean(a1); O.title(a1,'Where the wide-band error lives, and when it collapses')

sel5,Ar5=out[('fem_gp',5)]; sel4,Ar4=out[('fem',4)]; sel8,Ar8=out[('fem_gp',8)]
held=np.array([i for i in range(len(x)) if i not in set(sel5)])
per=np.median(np.abs(dB(Ar5[held])-dB(A[held])),axis=1)
iw=held[int(np.argmax(per))]
ref=dB(A).max()
a2.plot(FK,dB(A[iw])-ref,'-',color=O.INK,lw=2.6,alpha=.30,label='measured')
a2.plot(FK,dB(Ar4[iw])-ref,'--',color=O.ORANGE,lw=1.7,label='bare FEM, n = 4')
a2.plot(FK,dB(Ar5[iw])-ref,'-',color=O.GREEN,lw=1.7,alpha=.6,label='FEM + GP, n = 5')
a2.plot(FK,dB(Ar8[iw])-ref,'-',color=O.GREEN,lw=1.7,label='FEM + GP, n = 8')
a2.set_xlabel('frequency  (kHz)'); a2.set_ylabel('|Z|  (dB re max)')
a2.legend(fontsize=9.2,labelcolor=O.INK2,loc='lower left')
O.clean(a2)
O.title(a2,f'The single worst held-out position (x = {x[iw]:.1f} µm)')
fig.savefig('fig/d_wideband.png',dpi=170)
print('wrote fig/d_wideband.png')
