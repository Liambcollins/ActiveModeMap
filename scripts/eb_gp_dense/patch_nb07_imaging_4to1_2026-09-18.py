"""4:1 imaging frame + verified parameter application for notebook 07.

2026-09-18.  Two problems behind the streaky 30 um frame:

  1. Notebook and panel had drifted apart.  Cell 4 declares 6 um at 1.0 Hz, the
     frame actually run was 30 um at 1.6 Hz -- 96 um/s tip velocity against gains
     set for 12 um/s.  Nothing ever read the panel back, so the mismatch was
     invisible until the image looked wrong.
  2. `DriveAmplitude` is set once before the load series and never restored after
     the post-load-series tune (only `DriveFrequency` is), so the PFM drive at
     imaging time is whatever CanttuneFunc left behind.

This patch adds `setup_imaging()`, which pushes every parameter through PV() --
the applying path, unlike a raw assignment to root:...:Main:Variables: -- and then
verifies each one against get_gmv() before the scan is allowed to start.  It is
called immediately BEFORE take_image, not back in cell 9, because the load series
sits in between.

Frame: 4:1 at 6 um wide -> 6.00 x 1.50 um, 256 x 64 px, square 23.4 nm pixels.

Also folds in the useful parts of afm_laser_sweep_automation_v8: images get their
own "I"-prefixed base name so they never share a suffix counter with the "F"
force curves, BaseSuffix is reset before ARCheckSuffix(), and the drive frequency
is restored in a finally block so a failed scan cannot leave it parked.

Idempotent.  Run as:  python patch_nb07_imaging_4to1_2026-09-18.py <notebook.ipynb>
"""
import json
import re
import sys

MARKER = "IMG_BASENAME"

SETUP_SRC = '''# --- verified imaging setup (2026-09-18) ------------------------------------
# PV() writes AND applies; a raw assignment to root:...:Main:Variables: writes
# without applying. Everything below goes through PV() and is then read back
# from MasterVariables, because a silently-unapplied parameter is exactly what
# produced the streaky 30 um frame.
def setup_imaging(automation, igor, size_m, points, lines, rate_hz,
                  f_drive, drive_V, mode=None, tol=0.02):
    want = [("ScanSize", size_m), ("ScanPoints", points), ("ScanLines", lines),
            ("ScanRate", rate_hz), ("DriveFrequency", f_drive),
            ("DriveAmplitude", drive_V)]
    for k, v in want:
        igor.Execute(f\'PV("{k}", {v})\')
    if mode:
        automation.ex("LastScanPopup_0", "MasterPanel", 0, mode)
    time.sleep(1)

    gmv, bad = automation.get_gmv(), []
    for k, v in want:
        if k not in gmv:
            print(f"  {k:<16} not in MasterVariables - cannot verify")
            continue
        got = gmv[k]
        ok = abs(got - v) <= tol * max(abs(v), 1e-12)
        print(f"  {k:<16} asked {v:<12.6g} reads {got:<12.6g} "
              f"{\'ok\' if ok else \'** MISMATCH\'}")
        if not ok:
            bad.append(k)
    if bad:
        raise RuntimeError(f"panel did not take {bad} - not imaging on this")

    w_um = size_m * 1e6
    h_um = w_um * lines / points
    print(f"  frame {w_um:.2f} x {h_um:.2f} um  ({points} x {lines} px, "
          f"{size_m * 1e9 / points:.1f} nm/px)")
    print(f"  tip {2 * w_um * rate_hz:.0f} um/s, {lines / rate_hz:.0f} s/frame")
    print(f"  best achievable spot margin {0.5 * h_um:.2f} um "
          f"(find_domain_spots raises below min_margin_um = 0.5 um)")
    return gmv
'''

TAKE_IMAGE_SRC = '''# Same scan path notebook 00 uses: DownScan_0 on the master panel, then poll
# root:packages:MFP3D:OutWaves until the scan stops outputting.
def _scanning(igor):
    return bool(igor.DataFolder(r\'root:packages:MFP3D\').Wave(\'OutWaves\')
                .GetTextWavePointValue(0, 0))

def take_image(automation, base_filename, timeout_s=1800):
    automation.set_folder()
    igor.Execute(f\'root:Packages:MFP3D:Main:Variables:BaseName = "{base_filename}"\')
    igor.Execute(\'PV("BaseSuffix", 0000)\')
    igor.Execute(\'ARCheckSuffix()\')
    expected = automation._get_next_filename(base_filename, \'.ibw\')
    automation.ex(\'DownScan_0\', \'MasterPanel\')
    time.sleep(10)
    t0 = time.time()
    while _scanning(igor) and time.time() - t0 < timeout_s:
        time.sleep(2)
    if time.time() - t0 >= timeout_s:
        print(\'  WARNING: scan did not finish within the timeout\')
    time.sleep(3)
    return expected

# Apply and VERIFY the frame here, not back in the pre-load-series cell: the load
# series runs in between and every tune leaves its own DriveFrequency/Amplitude.
setup_imaging(inst.a, igor, SCAN_SIZE_M, SCAN_POINTS, SCAN_LINES,
              SCAN_RATE_HZ, f_cr, PFM_DRIVE_V)

# Images take their own "I" base name (v8 convention) so they never share a
# suffix counter with the "F" force curves or the optical .tif captures.
IMG_BASENAME = \'I\' + base_filename
try:
    img_path = take_image(inst.a, IMG_BASENAME)
finally:
    igor.Execute(f\'PV("DriveFrequency", {f_cr})\')

print(\'image saved:\', img_path, os.path.exists(img_path))
inst.a.withdraw()
print(\'channels:\', list_channels(img_path))
'''


def main(path):
    nb = json.load(open(path, encoding="utf-8"))
    cells = nb["cells"]

    def src(i):
        return "".join(cells[i]["source"])

    def setsrc(i, s):
        cells[i]["source"] = s.splitlines(keepends=True)

    if any(MARKER in src(i) for i in range(len(cells))):
        print("already patched - nothing to do")
        return 0

    # ---- cell 4: add the 4:1 aspect and an explicit SCAN_LINES ----
    par = next(i for i in range(len(cells)) if "SCAN_SIZE_M" in src(i))
    s = src(par)
    if "SCAN_LINES" not in s:
        s = re.sub(
            r"(SCAN_POINTS\s*=\s*\d+.*)$",
            r"\1\n"
            r"SCAN_ASPECT   = 4                     # width:height of the frame (4:1 strip)\n"
            r"SCAN_LINES    = SCAN_POINTS // SCAN_ASPECT   # 64 lines -> 1.50 um slow axis, square pixels",
            s, count=1, flags=re.M)
        setsrc(par, s)
        print(f"cell {par}: added SCAN_ASPECT / SCAN_LINES")

    # ---- replace the take_image cell, and put setup_imaging just before it ----
    ti = next(i for i in range(len(cells)) if "def take_image(" in src(i))
    setsrc(ti, SETUP_SRC + "\n" + TAKE_IMAGE_SRC)
    print(f"cell {ti}: replaced - setup_imaging() (4:1, with lines) + take_image + call")

    json.dump(nb, open(path, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
