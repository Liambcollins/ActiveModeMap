"""R4 frame list and calibration states. M350 = E/Q of the only valid two-domain bracket (350 nN, 02:17).
M700_cond = mean single-domain E/Q of the after_700 bracket evaluated at V_cpd(before) (assumes V_cpd unchanged)."""
import json, sys
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent; OUT = HERE.parent / "results/r3r4"
br = json.load(open(OUT / "r4_brackets.json")); b = {r["bracket"]: r for r in br["brackets"]}
M350 = b["before_350"]["M"]; M700 = float(np.mean([b["after_700"]["s1_M_at_Vref"], b["after_700"]["s2_M_at_Vref"]]))
M350_after = float(np.mean([b["after_350"]["s1_M_at_Vref"], b["after_350"]["s2_M_at_Vref"]]))
states = {"M350": dict(M=M350, Q_ref=b["before_350"]["Q"], note="valid bracket before imaging"),
          "M350_after_cond": dict(M=M350_after, Q_ref=b["after_350"]["Q"], note="after_350, single domain at V_ref"),
          "M700_cond": dict(M=M700, Q_ref=b["after_700"]["Q"], note="after_700, single domain at V_ref")}
R4 = "DomainsB_SCMPIT_R4"; bd = f"{R4}/52_image_be"
# InvOLS: the force curve taken at each (re)engage before the frame (FSCMPIT0495..0510)
be = [("reg_coarse64", 350, 0.0, 826.4, ["M350"]), ("be_0V", 350, 0.0, 826.4, ["M350"]), ("be_Vcpd", 350, 1.5366, 826.4, ["M350"]),
      ("be_0V_repeat", 350, 0.0, 826.4, ["M350", "M350_after_cond"]), ("be_0V_700nN", 700, 0.0, 846.5, ["M350", "M700_cond"]),
      ("reg2x_coarse64_700nN", 700, 0.0, 855.8, ["M350", "M700_cond"]), ("wtest_pm1_700nN", 700, 0.0, 838.9, ["M350", "M700_cond"]),
      ("wscale_pm1_700nN_0", 700, 0.0, 832.3, ["M350", "M700_cond"]), ("wscale_pm2_700nN_1", 700, 0.0, 832.3, ["M350", "M700_cond"]),
      ("wscale_pm4_700nN_2", 700, 0.0, 832.3, ["M350", "M700_cond"]), ("wscale_pm1_700nN_3", 700, 0.0, 832.3, ["M350", "M700_cond"]),
      ("bew4_0V_700nN_128px", 700, 0.0, 842.0, ["M350", "M700_cond"]), ("bew4_Vcpd_700nN_128px", 700, 1.5366, 835.1, ["M350", "M700_cond"])]
frames = [dict(kind="BE", name=n, path=f"{bd}/BE_{n}.hf5", states=st, load_nN=L, bias_V=V, invols=inv * 1e-9, vac=1.0) for n, L, V, inv, st in be]
qs = [("QImg0488", "blind-spot walk 1 (x~221)", 426), ("QImg0490", "blind-spot walk 2 (x~218)", 426), ("QImg0492", "blind-spot walk 3 (x~211)", 426),
      ("QImg0494", "blind-spot walk 4 (x~206)", 426), ("QImg0496", "blind-spot walk 5 (x~200)", 426), ("QImg0498", "blind-spot 256px (x~221)", 426),
      ("QImg0500", "qs20k_0V_initial 6um", 350), ("QImg0503", "qs20k_0V_after 6um", 350), ("QImg0505", "qs20k_0V_700nN 6um", 700),
      ("QImg0507", "qs20k_0V_12um", 350), ("QImg0509", "qs20k_0V_12um_700nN", 700),
      ("QS_after_final0001", "unlogged 09:16 12um (x~221, SP 0.447)", 411), ("QS_after_final0002", "unlogged 09:20", 411),
      ("QS_after_final0003", "unlogged 09:25", 411), ("QS_after_final0006", "unlogged 09:35 (SP 0.55)", 470)]
frames += [dict(kind="QS", name=f"{f} {d}", path=f"{R4}/{f}.ibw", state="-", load_nN=L, bias_V=0.0) for f, d, L in qs]
json.dump(dict(states=states, frames=frames), open(OUT / "r4_frame_calibration.json", "w"), indent=1)
print(json.dumps(states, indent=1))
