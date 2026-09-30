"""Imaging vs spectroscopy, same conventions on both sides:
QS image two-domain contrast vs survey two-domain |P_QS| (both NOT CPD-corrected) at the same lever position;
CR1 image contrast / survey E at that position vs the QS image. Positions in the lever frame (clamp from the
InvOLS ruler of each stage: SCM-PIT R1 survey 6.1 / images 17.3 um, R2 survey 24.1 / images 25.1 um)."""
import json, sys, os
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE.parent))
from fmmpaper import beimg
root = Path(sys.argv[1]); OUT = HERE.parent / "results/r3r4"
X = pd.read_csv(OUT / "xprobe_table.csv")
def surv(ds, x_stage, col):
    t = X[X.dataset == ds].sort_values("x_um"); return float(np.interp(x_stage, t.x_um, t[col]))
def img(path):
    q = beimg.load_qs(path); ph = np.radians(q["phase_deg"]); lab, _ = beimg.split_domains(ph); st = beimg.domain_stats(q["amp_m"], ph, lab)
    return st["contrast"] * 1e12 / q["drive_V"], st[0]["amp_med"] * 1e12 / q["drive_V"], st[1]["amp_med"] * 1e12 / q["drive_V"]
rows = []
cases = [("SCM-PIT R2 (500 nN)", "A", 154.9, 25.1, 24.1, "DomainsB_SCMPIT_R2/50_quant_imaging", "QImg0708", "QImg0711"),
         ("SCM-PIT R2 (500 nN)", "B", 232.0, 25.1, 24.1, "DomainsB_SCMPIT_R2/50_quant_imaging", "QImg0714", "QImg0716"),
         ("SCM-PIT R1 (500 nN)", "A", 154.9, 17.3, 6.1, "DomainsB_SCMPIT/50_quant_imaging", "QImg0573", "QImg0575"),
         ("SCM-PIT R1 (500 nN)", "B", 232.0, 17.3, 6.1, "DomainsB_SCMPIT/50_quant_imaging", "QImg0578", "QImg0580")]
for ds, pos, xs_img, x0i, x0s, fold, fqs, fcr in cases:
    lever = xs_img - x0i; xs = lever + x0s
    cq, q1, q2 = img(root / fold / f"{fqs}.ibw"); cc, c1, c2 = img(root / fold / f"{fcr}.ibw")
    Pq = surv(ds, xs, "d33_qs_raw"); E = surv(ds, xs, "E")
    rows.append(dict(probe=ds, where=f"{pos} (lever {lever:.0f} um)", kind="QS image vs survey QS", image=cq, spectroscopy=Pq, ratio=cq / Pq,
                     note=f"domains {q1:.2f} / {q2:.2f}"))
    rows.append(dict(probe=ds, where=f"{pos} (lever {lever:.0f} um)", kind="CR1 image (fixed f, at null) / survey E vs QS image", image=cc / E, spectroscopy=cq, ratio=cc / E / cq,
                     note=f"CR1 contrast {cc:.0f} pm/V, survey E {E:.0f}"))
# Multi75-G (R3/R4): from the frame tables
r4 = pd.read_csv(OUT / "r4_frames.csv"); r3 = pd.read_csv(OUT / "r3_frames.csv"); br = json.load(open(OUT / "r4_brackets.json"))["brackets"]
qs350 = r4[r4.frame.str.contains("QImg0500|QImg0503")].d33_best.mean(); qs700 = r4[r4.frame.str.contains("QImg0505")].d33_best.iloc[0]
rows.append(dict(probe="Multi75-G R4 (350 nN)", where="~159 um", kind="QS image vs bracket QS", image=qs350, spectroscopy=br[0]["d33_qs"], ratio=qs350 / br[0]["d33_qs"], note="0 V frames before/after BE"))
b700 = np.mean([br[2]["s1_d33_qs_at_Vref"], br[2]["s2_d33_qs_at_Vref"]])
rows.append(dict(probe="Multi75-G R4 (700 nN)", where="~159 um", kind="QS image vs bracket QS", image=qs700, spectroscopy=b700, ratio=qs700 / b700, note="bracket conditional (single domain at V_cpd)"))
be700 = r4[(r4.frame == "be_0V_700nN") & (r4.state == "M350")].d33_best.iloc[0]
rows.append(dict(probe="Multi75-G R4 (700 nN)", where="~159 um", kind="BE image (E from bracket) vs QS image", image=be700, spectroscopy=qs700, ratio=be700 / qs700, note="±1 window, 256 px, 8.2 ms"))
bew = r4[(r4.frame == "bew4_Vcpd_700nN_128px") & (r4.state == "M350")].d33_best.iloc[0]
rows.append(dict(probe="Multi75-G R4 (700 nN)", where="~159 um", kind="BE image (E from bracket) vs QS image", image=bew, spectroscopy=qs700, ratio=bew / qs700, note="±4 window, 128 px, 4.1 ms, at V_cpd"))
be3 = r3[(r3.kind == "BE") & (r3.state == "survey_E")].d33_best
rows.append(dict(probe="Multi75-G R3 (350 nN)", where="~108 um", kind="BE image (survey M) vs survey QS, corrected", image=be3.mean(), spectroscopy=7.09, ratio=be3.mean() / 7.09, note=f"4 frames {be3.min():.2f}-{be3.max():.2f}"))
q3 = r3[r3.frame.str.startswith("A_qs")].d33_best.iloc[0]; s3 = surv("Multi75-G R3 (350 nN)", 108.4, "d33_qs_raw")   # survey on the InvOLS axis ~ labels
rows.append(dict(probe="Multi75-G R3 (350 nN)", where="~108 um", kind="QS image vs survey QS", image=q3, spectroscopy=s3, ratio=q3 / s3, note="image 30 min after the survey"))
# soft probe: tip-end image series (InvOLS 505 nm/V = tip stop), 30 kHz reference
cq, q1, q2 = img(root / "DomainsBPPPCONTAU/06_bias_image_series/BiasImg0842.ibw")
rows.append(dict(probe="PPP-CONTAu (~100 nN)", where="tip end", kind="QS image vs survey QS at the tip (15 / 250 nN)", image=cq,
                 spectroscopy=float(np.mean([surv('PPP-CONTAu (15 nN)', 445, 'd33_qs_raw'), surv('PPP-CONTAu (250 nN)', 445, 'd33_qs_raw')])), ratio=np.nan,
                 note=f"survey tip: 15 nN {surv('PPP-CONTAu (15 nN)', 445, 'd33_qs_raw'):.2f}, 250 nN {surv('PPP-CONTAu (250 nN)', 445, 'd33_qs_raw'):.2f}"))
T = pd.DataFrame(rows); T["ratio"] = T.image / T.spectroscopy
T.to_csv(OUT / "xprobe_imaging.csv", index=False); pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 60); print(T.round(3).to_string())
