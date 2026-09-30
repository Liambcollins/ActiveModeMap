from igor2 import binarywave as bw
import numpy as np
def note_kv(w):
    note=w['note'].decode('latin1',errors='ignore')
    return dict((l.split(':',1)[0].strip(), l.split(':',1)[1].strip()) for l in note.replace('\r','\n').split('\n') if ':' in l)
def invols_fc(path, lo=0.3, hi=0.9):
    """InvOLS from the contact part of a saved force curve: Defl (m) was written with the note InvOLS,
    so V = Defl/InvOLS_note and InvOLS = dZ/dV = InvOLS_note * dZ/dDefl."""
    w=bw.load(path)['wave']; D=w['wData'].astype(float); kv=note_kv(w)
    inv0=float(kv['InvOLS']); z=D[:,4]; de=D[:,1]
    idx=[int(v) for v in kv['Indexes'].split(',')]
    out={}
    for name,(a,b) in (('ext',(idx[0],idx[1])),('ret',(idx[1],idx[2]))):
        dd=de[a:b]; zz=z[a:b]
        base=np.median(dd[:len(dd)//5]) if name=='ext' else np.median(dd[-len(dd)//5:])
        span=dd.max()-base; m=(dd>base+lo*span)&(dd<base+hi*span)
        p=np.polyfit(dd[m],zz[m],1) if m.sum()>8 else [np.nan]
        out[name]=abs(p[0])*inv0
    return dict(invols=out['ext'], invols_ret=out['ret'], note_invols=inv0, defl_max_nm=float((de.max()-np.median(de[:200]))*1e9), kv=kv)
