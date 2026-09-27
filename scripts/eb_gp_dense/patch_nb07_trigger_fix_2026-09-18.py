"""Fix the calibration force-curve trigger in notebooks/07_pfm_image_two_domain_bias.ipynb.

2026-09-18: the load series died at positions 2-4 (x = 392.1 / 339.3 / 286.4) with

    ERROR loading force curve: [Errno 2] No such file or directory: '...FDomBias0006.ibw'
    cond 0 (+0.00 V, 50 nN) FAILED: RuntimeError: calibrate first (InvOLS / spring constant unknown)

Cause, not the filename counter: `AFMLaserSweepAutomation.measure_invols` sets the
force-curve trigger point from the panel's CURRENT DeflectionSetpointVolts.  The
loads run ascending, so every position ends on the largest load -- 450 nN, which at
the free end (InvOLS 5.1e-7, k 0.148) is a ~5.96 V setpoint.  The next position's
calibration curve then triggers at a deflection the ramp never reaches, Igor saves no
.ibw, `measure_invols` swallows the exception and returns None, and every `set_load`
at that position raises "calibrate first".  Position 1 worked only because its
calibration ran before any set_load, while the panel still held the engage setpoint.

Idempotent.  Run as:  python patch_nb07_trigger_fix_2026-09-18.py <notebook.ipynb>
"""
import json
import sys

MARKER = "2026-09-18 fix: calibration force-curve trigger"

CELL_SRC = '''# --- 2026-09-18 fix: calibration force-curve trigger -------------------------
# measure_invols() sets the force-curve trigger from the panel's CURRENT
# DeflectionSetpointVolts. Loads run ascending, so each position ends on the
# largest load (450 nN ~= 5.96 V at the free end) and the NEXT position's
# calibration curve triggers at a deflection the ramp never reaches: Igor saves
# no .ibw, measure_invols returns None, and every set_load there then raises
# "calibrate first". Observed at x = 392.1 / 339.3 / 286.4 after a clean pos 1.
#
# Fix: derive the trigger from the last known InvOLS and the LIGHTEST load in the
# series. A fixed voltage will not do -- 1 V is ~75 nN at the free end but
# ~1.3 uN at x = 75, where InvOLS is 9.1e-6 m/V.
#
# Patches the classes, so it also covers the bias survey (S7) and cluster (S7b).
import time as _time
import activemodemap.asylum as _A

CAL_TRIGGER_LOAD_NN = 50.0          # lightest load in LOADS_NN

if not getattr(_A.AFMLaserSweepAutomation.measure_invols, "_trigger_fix", False):
    _orig_measure_invols = _A.AFMLaserSweepAutomation.measure_invols

    def _measure_invols_safe(self):
        inv = getattr(self, "last_invols", None)
        if inv:
            k = self.get_gmv()["SpringConstant"]
            v = (CAL_TRIGGER_LOAD_NN * 1e-9) / k / inv
            self.igor.Execute(f\'PV("DeflectionSetpointVolts", {v})\')
            _time.sleep(0.3)
            print(f"      cal trigger {v:.3f} V "
                  f"({CAL_TRIGGER_LOAD_NN:.0f} nN at InvOLS {inv:.2e} m/V)")
        else:
            print("      WARNING: no last_invols -- calibration will trigger on "
                  "whatever the MasterPanel currently holds. Check that setpoint "
                  "before continuing; a stale 450 nN value is what broke this run.")
        return _orig_measure_invols(self)

    _measure_invols_safe._trigger_fix = True
    _A.AFMLaserSweepAutomation.measure_invols = _measure_invols_safe

# Fail at the source. A dead force curve should not surface as four
# "calibrate first" errors one layer up, three positions in a row.
if not getattr(_A.AsylumInstrument._goto_and_prepare, "_invols_guard", False):
    _orig_goto_and_prepare = _A.AsylumInstrument._goto_and_prepare

    def _goto_and_prepare_checked(self, x_um, capture_image=True):
        label = _orig_goto_and_prepare(self, x_um, capture_image)
        if self._invols is None:
            raise RuntimeError(
                f"InvOLS calibration returned None at x={x_um:.1f} um -- the force "
                "curve did not save. Check the MasterPanel deflection trigger and "
                "that AutoWedge left the photodiode on scale.")
        return label

    _goto_and_prepare_checked._invols_guard = True
    _A.AsylumInstrument._goto_and_prepare = _goto_and_prepare_checked

print(f"calibration-trigger fix armed: {CAL_TRIGGER_LOAD_NN:.0f} nN trigger, "
      "InvOLS guard on")
'''

MD_SRC = '''### Calibration-trigger fix (2026-09-18)

The cell below must run before any `run_series` call. It repairs the interaction
between ascending loads and the per-position InvOLS calibration that stopped the
first load-series attempt at position 2. It patches the classes, so one run covers
the load series, the bias survey and the cluster.
'''


def main(path):
    nb = json.load(open(path, encoding="utf-8"))
    cells = nb["cells"]

    def src(i):
        return "".join(cells[i]["source"])

    if any(MARKER in src(i) for i in range(len(cells))):
        print("already patched - nothing to do")
        return 0

    target = None
    for i, c in enumerate(cells):
        if c["cell_type"] == "code" and "run_series(" in src(i):
            target = i
            break
    if target is None:
        raise KeyError("no run_series cell found")

    md = {"cell_type": "markdown", "metadata": {},
          "source": MD_SRC.splitlines(keepends=True)}
    code = {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": CELL_SRC.splitlines(keepends=True)}
    cells[target:target] = [md, code]

    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    print(f"inserted markdown + code cell at index {target} "
          f"(before the run_series cell, now at {target + 2})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
