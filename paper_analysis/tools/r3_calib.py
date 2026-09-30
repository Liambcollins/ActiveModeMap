"""R3 frame calibration: M = E/Q at the imaging position. The laser sat at ~108 um (InvOLS 1584 nm/V), not the
96.3 um label, so the survey E and Q are interpolated to x_equiv; three E variants are carried:
survey E (as measured), E_mixed = |P_cr| / |P_qs corrected for the domain CPD| (the combination an image sees),
and the free-air + sparse model E. The agent's M = 500/267 is kept for comparison."""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "results/r3r4"
S = json.load(open(OUT / "r3_survey.json")); t = pd.DataFrame(S["table"])
pos = pd.read_csv(OUT / "positions.csv"); sv = pos[pos.stage == "survey"]
t["x_equiv"] = [sv[np.isclose(sv.x_label, x)].x_equiv.iloc[0] for x in t.x_um]
img = pos[pos.stage.str.startswith("imaging")]; x_img = float(img.x_equiv.mean()); inv_img = float(img.invols_nm.mean()) * 1e-9
t = t.sort_values("x_equiv")
Em = t.E * t.d33_qs / t.d33_qs_corr_no232
def at(y): return float(np.exp(np.interp(x_img, t.x_equiv, np.log(y))))
E_s, E_mix, Q_s = at(t.E), at(Em), at(t.Q)
mod = json.load(open(OUT / "r3_E_model_vs_survey.json"))["calfa_mu0.026"]
E_mod = float(np.exp(np.interp(x_img, mod["x_eval"], np.log(mod["E_eval"])))); Q_mod = float(np.interp(x_img, mod["x_eval"], mod["Q_eval"]))
states = {"survey_E": dict(M=E_s / Q_s, Q_ref=267.1, E_at_Qsurvey=E_s, Q_survey=Q_s, note="survey E/Q interpolated to x_equiv; Q_ref = tune Q at imaging"),
          "mixed_E": dict(M=E_mix / Q_s, Q_ref=267.1, note="|P_cr| / |P_qs, CPD-corrected|"),
          "model_E": dict(M=E_mod / Q_mod, Q_ref=267.1, note="calibrate_freeair (mu 0.026) at x_equiv"),
          "agent_500_267": dict(M=500 / 267.15, Q_ref=267.1, note="README: survey E at 96.3 um (label) with the tune Q")}
R3 = "DomainsB_SCMPIT_R3"
be = [("A_be_0V", 0.0), ("A_be_Vhalf", 2.547), ("A_be_Vcpd", 5.0935), ("A_be_0V_repeat", 0.0)]
frames = [dict(kind="BE", name=n, path=f"{R3}/52_image_be/BE_{n}.hf5", states=list(states), load_nN=350, bias_V=v, invols=inv_img, vac=1.0) for n, v in be]
frames += [dict(kind="QS", name="A_qs20k_0V (16:19)", path=f"{R3}/QImg0079.ibw", state="-", load_nN=350, bias_V=0.0),
           dict(kind="QS", name="start0000 (13:44) @459", path=f"{R3}/quasi_d33_start0000.ibw", state="-", load_nN=426, bias_V=0.0, invols_true=458.9e-9),
           dict(kind="QS", name="start0001 (13:48) @459", path=f"{R3}/quasi_d33_start0001.ibw", state="-", load_nN=426, bias_V=0.0, invols_true=458.9e-9),
           dict(kind="QS", name="start0000 (13:44) @490", path=f"{R3}/quasi_d33_start0000.ibw", state="-", load_nN=426, bias_V=0.0, invols_true=489.8e-9)]
json.dump(dict(x_img_equiv=x_img, invols_img=inv_img, states=states, frames=frames), open(OUT / "r3_frame_calibration.json", "w"), indent=1)
print(json.dumps(dict(x_img_equiv=x_img, invols=inv_img, states=states), indent=1))
