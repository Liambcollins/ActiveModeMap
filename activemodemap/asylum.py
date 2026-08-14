"""Asylum Research / Igor Pro instrument control for ActiveModeMap.

The `AFMLaserSweepAutomation` class is adapted (faithfully) from the user's
`afm_laser_sweep_automation_v4` notebook — it drives an Asylum/Cypher system
through the Igor Pro COM bridge: move the detection laser, capture an optical
image (for spot-position verification), run AutoWedge + InvOLS calibration,
engage at a set load, and tune the contact resonance (saving Frequency / Phase /
Amplitude).

`AsylumInstrument` wraps that class in the `Instrument.measure_at(x_um)`
interface used by the online active-learning loop, so the same loop code runs on
the microscope and against the VirtualInstrument.

WINDOWS ONLY: importing this module needs `win32com` and `igor2`, present on the
instrument PC. On any other machine, import `online.VirtualInstrument` instead.
(`read_tune_txt` is pure-Python and importable anywhere — see below.)

IMPORTANT — fixed tune window. For mode-shape mapping the tune must cover the
whole resonance + antiresonance structure and must NOT auto-recenter per
position. Set a fixed tune range in Igor before running the loop (the same way
you set it for the dense sweep); `measure_at` reads whatever window Igor is
configured for and never changes it. Keep the window TIGHT around the band you
are mapping — see `AsylumInstrument(analysis_band_Hz=...)` and the note on
frequency resolution in the README.
"""

from __future__ import annotations

import os
import re
import time
import numpy as np

from .online import Instrument


# --------------------------------------------------------------------------- #
#  Igor tune-file reader (pure Python — no Igor / Windows needed)             #
# --------------------------------------------------------------------------- #
def read_tune_txt(path):
    """Parse an Igor `Save/T` text wave file written by `save_tune`.

    Igor writes these with carriage returns (\\r) rather than newlines, a
    `WAVES` header line, then the numeric block delimited by BEGIN / END, then
    trailing `X ...` command lines. Returns the full spectrum as
    ``{'frequency', 'phase', 'amplitude'}`` (phase in degrees, as saved).

    This is the authoritative reader: the file that Igor saves always contains
    the complete tune, so `tune_eigenmode` prefers it over the COM bridge.
    """
    txt = open(path, "rb").read().decode("latin-1").replace("\r\n", "\n")
    txt = txt.replace("\r", "\n")
    lines = txt.split("\n")

    # column order comes from the WAVES header (save_tune writes Freq/Phase/Amp)
    cols = ["Frequency", "Phase", "Amp"]
    for ln in lines[:10]:
        if ln.startswith("WAVES"):
            hdr = [c.strip() for c in ln.split("\t")[1:] if c.strip()]
            if len(hdr) >= 3:
                cols = hdr
            break

    rows, started = [], False
    for ln in lines:
        s = ln.strip()
        if s == "BEGIN":
            started = True
            continue
        if s == "END":
            break
        if started and s:
            try:
                rows.append([float(v) for v in s.split("\t")])
            except ValueError:
                continue
    if not rows:
        raise ValueError(f"no numeric data found in {path}")
    a = np.asarray(rows, float)

    def pick(name, default_i):
        for i, c in enumerate(cols):
            if c.lower().startswith(name):
                return a[:, i]
        return a[:, default_i]

    return {"frequency": pick("freq", 0),
            "phase": pick("phase", 1),
            "amplitude": pick("amp", 2)}


def tune_to_complex(tune_data, band_Hz=None, drop_nonfinite=True):
    """(freq, Z) from a tune dict, optionally cropped to ``band_Hz=(lo, hi)``.

    Phase is stored in degrees by Igor; Z = amplitude * exp(i * phase).

    Igor commonly leaves the Amp/Phase waves NaN outside the range actually
    swept, while the Frequency wave stays fully populated. Those NaNs are poison
    downstream: `np.linalg.lstsq` propagates them into every reconstructed
    column, and a single NaN makes `np.abs(Zrec).max()` NaN, which silently
    blanks the uncertainty plot and leaves the map painting only part of its
    axes. `drop_nonfinite` removes them here, at the boundary, so nothing
    downstream has to cope.
    """
    f = np.asarray(tune_data["frequency"], float)
    amp = np.asarray(tune_data["amplitude"], float)
    ph = np.asarray(tune_data["phase"], float)
    if drop_nonfinite:
        good = np.isfinite(f) & np.isfinite(amp) & np.isfinite(ph)
        if not good.all():
            n_bad = int((~good).sum())
            kept = f[good]
            print(f"    dropped {n_bad} non-finite tune points "
                  f"({100 * n_bad / f.size:.0f}% of the wave); keeping "
                  f"{kept.size} points, {kept.min() / 1e3:.1f}-"
                  f"{kept.max() / 1e3:.1f} kHz")
            f, amp, ph = f[good], amp[good], ph[good]
        if f.size == 0:
            raise ValueError("tune contains no finite points")
    Z = amp * np.exp(1j * np.deg2rad(ph))
    if band_Hz is not None:
        lo, hi = float(band_Hz[0]), float(band_Hz[1])
        m = (f >= lo) & (f <= hi)
        if m.sum() < 8:
            raise ValueError(
                f"analysis_band_Hz=({lo:.0f}, {hi:.0f}) keeps only {int(m.sum())} "
                f"of {f.size} tune points (window is {f[0]:.0f}-{f[-1]:.0f} Hz). "
                "Check the band against the tune range set in Igor.")
        f, Z = f[m], Z[m]
    return f, Z


# --------------------------------------------------------------------------- #
#  Automation class (adapted from afm_laser_sweep_automation_v4)              #
# --------------------------------------------------------------------------- #
def _note_value(note, key):
    """Numeric value of `key` from an Igor wave note.

    Replaces `float(str(note).split('rInvOLS: ')[1].split('\\')[0])`, which only
    ever worked by accident: `str(bytes)` renders the note's CR line separators
    as a literal backslash-r, so the leading 'r' in 'rInvOLS: ' was matching the
    line break before 'InvOLS: '. It returns the right number on these files, but
    it silently grabs the wrong line the moment Igor writes LF instead of CR, or
    a key ending in 'rInvOLS' appears earlier in the note.

    Splits on real line breaks and requires an exact key match, so 'InvOLS' can
    never pick up 'AmpInvOLS' or 'Amp2InvOLS'.
    """
    if isinstance(note, (bytes, bytearray)):
        note = note.decode('latin-1', 'replace')
    else:
        note = str(note)
    for line in note.replace('\r\n', '\n').replace('\r', '\n').split('\n'):
        k, sep, v = line.partition(':')
        if sep and k.strip() == key:
            try:
                return float(v.strip())
            except ValueError:
                break
    raise KeyError(
        f"key {key!r} not found as a numeric entry in the wave note "
        f"({len(note)} chars). Do not guess a calibration — check the note.")


class AFMLaserSweepAutomation:
    """Automated laser positioning, calibration, engage, and tune on Asylum/Igor."""

    #: below this many points a "spectrum" is certainly a truncated read
    MIN_TUNE_POINTS = 32

    def __init__(self, igor, file_loc, base_filename, log_filename):
        self.igor = igor
        self.file_loc = file_loc
        self.base_filename = base_filename
        self.log_filename = log_filename
        self.load_force_setpoint = 0.5
        self.tune_settling_time = 2.0
        #: hard cap on how long to wait for a tune to finish (s)
        self.tune_timeout_s = 180.0
        #: poll interval while waiting for a tune (s)
        self.tune_poll_s = 0.4
        #: consecutive unchanged polls that count as "finished"
        #: consecutive unchanged polls that count as "finished". At the default
        #: 0.4 s poll this is a ~2 s quiet window, which is long enough that a
        #: momentary lull mid-sweep is not mistaken for completion.
        self.tune_stable_polls = 5
        #: ABSOLUTE plausibility window (m/V) for a freshly measured InvOLS.
        #: Only a measurement outside this is refused. This used to be
        #: (4e-8, 10e-7), and the 1.0e-6 ceiling was BELOW the true InvOLS over
        #: the base half of a Multi75E-G -- it reaches 2.5e-6 at x ~ 85 um. Every
        #: such value was silently dropped, so on Demo6Fullgrid the panel InvOLS
        #: froze at 9.9613e-7 for the last ~170 positions of the dense sweep and
        #: every physical unit Igor derived there (the saved Deflection channel
        #: in metres, DeflectionSetpointNewtons, the force display) was wrong by
        #: up to 2.5x. Keep this wide: it is a sanity check, not a calibration.
        self.invols_bounds = (1e-8, 1e-5)
        #: Warn (but still apply) when a new InvOLS differs from the last
        #: accepted one by more than this factor. Adjacent 0.5 um steps change it
        #: by well under 1%, so a big jump means a bad force curve or a move that
        #: did not land -- worth seeing, not worth discarding.
        self.invols_warn_ratio = 3.0
        #: Last accepted InvOLS, and a log of rejected measurements, so a run can
        #: be audited afterwards instead of failing silently.
        self.last_invols = None
        self.invols_rejected = []
        #: Whether to run AutoWedge before each InvOLS measurement. This flag was
        #: previously ignored -- `_goto_and_prepare` called AutoWedge
        #: unconditionally, so every run to date (Demo6 included) autowedged at
        #: every position despite this defaulting to False. It is now honoured;
        #: the default is True to match what actually ran, so results stay
        #: comparable with earlier data. Set it False to skip.
        self.use_autowedge = True
        self.autowedge_pause = 15.0
        self.eigenmode_center_freq = 70000
        #: band (Hz) used when reporting f_res / Q; None = whole tune window
        self.resonance_band_Hz = None
        #: sample-spot move (GoToSpot) settings
        self.spot_timeout_s = 90.0
        self.spot_poll_s = 0.5
        self.spot_stable_polls = 4
        self.spot_stable_tol_V = 2e-3
        self.results = []

    # -- low-level ------------------------------------------------------- #
    def ex(self, variable="", panel="", val=0, string="", verbose=False):
        line = ("print " if verbose else "") + \
            f'ARExecuteControl("{variable}", "{panel}", {val}, "{string}")'
        self.igor.Execute(line)
        return line

    @staticmethod
    def _wave_npnts(wave):
        """Number of points in a 1-D Igor wave read over COM.

        Igor's COM ``Wave.GetDimensions()`` returns the DIMENSION COUNT first
        and the row count second — i.e. ``(nDims, rows, cols, layers, chunks)``.
        For a 1-D wave element 0 is 2, so the long-standing
        ``GetDimensions()[0]`` silently returned **only the first two points of
        every tune**. `get_gmv` has always (correctly) used element 1.

        We take the largest reported dimension, which is the row count for the
        1-D Frequency / Phase / Amp waves and is immune to further changes in
        the tuple layout.
        """
        dims = wave.GetDimensions()
        try:
            vals = [int(v) for v in dims]
        except TypeError:                     # single scalar
            vals = [int(dims)]
        return max(vals)

    def get_gmv(self):
        data = self.igor.DataFolder(r"root:packages:MFP3D:Main:Variables").Wave("MasterVariablesWave")
        dim = data.GetDimensions()[1]
        return {data.DimensionLabel(0, i, 0): data.GetNumericWavePointValue(i)
                for i in range(dim)}

    def set_folder(self):
        loc = self.file_loc.split('\\', 1)
        igorloc = loc[0] + loc[1].replace('\\', ':')
        if igorloc[-1] != ':':
            igorloc += ':'
        self.igor.Execute(f'root:Packages:MFP3D:Main:Strings:GlobalStrings[20] = "{igorloc}"')
        self.igor.Execute(f'InsertNewPathInHistory("{igorloc}")')
        self.igor.Execute(f'root:Packages:MFP3D:Main:Strings:GlobalStrings[18] = "{igorloc}"')
        self.igor.Execute(f'NewPath/O SaveImage "{igorloc}"')
        self.igor.Execute(f'NewPath/O SaveForce "{igorloc}"')

    def windows2igor(self, path):
        parts = path.split('\\', 1)
        igorp = (parts[0] + parts[1].replace('\\', ':')) if len(parts) == 2 \
            else path.replace('\\', ':')
        return igorp if igorp.endswith(':') else igorp + ':'

    def _get_next_filename(self, base_name, extension):
        pattern = re.compile(rf"^{re.escape(base_name)}(\d{{4}})\..+$")
        mx = -1
        for fn in os.listdir(self.file_loc):
            m = pattern.match(fn)
            if m:
                mx = max(mx, int(m.group(1)))
        return os.path.join(self.file_loc, f"{base_name}{mx + 1:04d}{extension}")

    # -- laser / engage -------------------------------------------------- #
    def do_ld_move(self, microns_x, microns_y):
        self.igor.Execute(f'DoLDMove({microns_x}, {microns_y})')
        time.sleep(0.5)

    def move_laser_to_position(self, x_um, y_um, relative=True):
        if relative:
            self.do_ld_move(x_um, y_um)
        else:
            self.igor.Execute("save_laser_position_in_igor()")
            cx = self.igor.DataFolder(r"root").Wave("LDX_pos").GetNumericWavePointValue(0)
            cy = self.igor.DataFolder(r"root").Wave("LDX_pos").GetNumericWavePointValue(1)
            self.do_ld_move((x_um * 10 - cx) / 10, (y_um * 10 - cy) / 10)

    def capture_optical_image(self, position_label=""):
        self.set_folder()
        self.igor.Execute('ARVideoButtonFunc("ARVCapture")')
        time.sleep(1)
        return "captured"

    def simple_engage(self, wait_time=5):
        self.igor.Execute('SimpleEngageMe("")')
        time.sleep(wait_time)
        return True

    def withdraw(self):
        self.igor.Execute('DoScanFunc("StopEngageButton")')
        time.sleep(2)

    def do_autowedge(self):
        self.igor.Execute("AutoWedge()")
        time.sleep(self.autowedge_pause)

    def measure_invols(self):
        from igor2 import binarywave
        self.set_folder()
        os.chdir(self.file_loc)
        gmv = self.get_gmv()
        self.ex("TriggerChannelPopup_1", "MasterPanel", 0, "DeflVolts")
        self.ex("TriggerPointSetVar_1", "MasterPanel", gmv["DeflectionSetpointVolts"])
        note = 'F'
        self.igor.Execute(f'root:Packages:MFP3D:Main:Variables:BaseName = "{note}{self.base_filename}"')
        self.igor.Execute('PV("BaseSuffix", 0000)')
        self.igor.Execute('ARCheckSuffix()')
        fname = self._get_next_filename(f'{note}{self.base_filename}', '.ibw')
        self.ex("SingleForce_1", "MasterPanel", 1)
        time.sleep(8)
        try:
            file = binarywave.load(fname)
            z = file['wave']['wData'][:, 4]
            defl = file['wave']['wData'][:, 1]
            max_i = np.argmax(defl)
            engage_i = np.argmin(defl[:max_i])
            curr = _note_value(file['wave']['note'], 'InvOLS')
            deflV = defl / curr
            di = max_i - engage_i
            i1 = int(engage_i + 0.25 * di); i2 = int(max_i - 0.25 * di)
            invols = (z[i2] - z[i1]) / (deflV[i2] - deflV[i1])
            lo, hi = self.invols_bounds
            if lo < invols < hi:
                prev = self.last_invols
                if prev and (invols / prev > self.invols_warn_ratio
                             or prev / invols > self.invols_warn_ratio):
                    print(f"  NOTE: InvOLS jumped {prev:.3e} -> {invols:.3e} m/V "
                          f"({invols / prev:.2f}x). Applying it, but check the "
                          "force curve and that the laser move landed.")
                self.ex('InvOLSSetVar_1', 'MasterPanel', invols)
                self.last_invols = invols
            else:
                # Loud, and recorded. Silently skipping the write-back is what
                # left half of Demo6Fullgrid on a stale calibration.
                self.invols_rejected.append(invols)
                print(f"  WARNING: measured InvOLS {invols:.3e} m/V is outside "
                      f"invols_bounds ({lo:.1e}, {hi:.1e}) — NOT written to the "
                      "MasterPanel, which is therefore now STALE. The returned "
                      "value is still used for the load setpoint. Widen "
                      "invols_bounds if this value is real.")
            return invols
        except Exception as e:
            print(f"  ERROR loading force curve: {e}")
            return None

    # -- sample spots (GoToSpot / Force panel pick-spot) ------------------- #
    def read_xy_sensor(self):
        """(XSensor, YSensor) in volts, read through a scratch wave.

        There is no direct COM call for td_ReadValue, so Execute writes both
        sensors into a small wave and we read the points back — the same pattern
        the laser-position code uses with LDX_pos.
        """
        self.igor.Execute('Make/O/N=2 root:AMM_XY')
        self.igor.Execute('root:AMM_XY[0] = td_ReadValue("XSensor")')
        self.igor.Execute('root:AMM_XY[1] = td_ReadValue("YSensor")')
        w = self.igor.DataFolder(r"root").Wave("AMM_XY")
        return (float(w.GetNumericWavePointValue(0)),
                float(w.GetNumericWavePointValue(1)))

    def goto_spot(self, spot_number, wait=True, verbose=False):
        """Drive the SAMPLE (scanner) to a spot marked on the current image.

        Spots are the ones picked on an image via the Force panel — they live in
        root:Packages:MFP3D:Force:SpotX/SpotY, and **index 0 is the scan origin;
        user-marked spots start at 1**. `GoToSpot()` itself does the offset,
        scan-angle and LVDT arithmetic and honours GoThereWithdraw, so we only
        select the spot and start it. Requires the image window with the marked
        spots to be OPEN in Igor, or GoToSpot raises an alert and does nothing.

        This moves the sample under the tip. It is completely independent of the
        detection-laser position along the cantilever (DoLDMove), which is
        untouched.

        Completion is detected by polling the X/Y sensors until they are stable
        within `spot_stable_tol_V` for `spot_stable_polls` polls — target-free,
        so it works whichever internal branch GoToSpot takes (direct ramp, or
        withdraw-then-ramp via its callback). Returns True when the scanner
        settled, False on timeout.
        """
        spot_number = int(spot_number)
        if spot_number < 1:
            raise ValueError(
                "spot_number must be >= 1: index 0 is the scan origin, "
                "user-marked spots start at 1")
        self.igor.Execute(f'PV("ForceSpotNumber", {spot_number})')
        time.sleep(0.2)
        self.igor.Execute('GoToSpot()')
        if not wait:
            return None
        t0 = time.time()
        try:
            prev = self.read_xy_sensor()
        except Exception as e:
            print(f"    WARNING: cannot read XY sensors ({e}); "
                  "using a fixed 10 s wait for the spot move")
            time.sleep(10.0)
            return False
        moved, stable = False, 0
        while time.time() - t0 < self.spot_timeout_s:
            time.sleep(self.spot_poll_s)
            cur = self.read_xy_sensor()
            d = max(abs(cur[0] - prev[0]), abs(cur[1] - prev[1]))
            if d > self.spot_stable_tol_V:
                moved, stable = True, 0
            else:
                stable += 1
                # If it never moved, either the ramp has not started yet (keep
                # waiting a little) or we were already at the spot (accept).
                if stable >= self.spot_stable_polls and (
                        moved or time.time() - t0 > 5.0):
                    if verbose:
                        print(f"      spot {spot_number} reached after "
                              f"{time.time() - t0:.1f} s "
                              f"(XY = {cur[0]:+.4f}, {cur[1]:+.4f} V)"
                              + ("" if moved else "  [no motion — already there?]"))
                    time.sleep(0.5)
                    return True
            prev = cur
        print(f"    WARNING: scanner did not settle within "
              f"{self.spot_timeout_s:.0f} s of GoToSpot (spot_timeout_s)")
        return False

    # -- tune ------------------------------------------------------------ #
    def _tune_signature(self, n_probe=48):
        """Fingerprint of the current tune waves, for change detection.

        One coarse pass over the Amp wave gives both signals at no extra cost:

        * **how many probed points are finite** -- Igor NaN-fills the part of the
          tune it has not swept yet, so this count rises monotonically with sweep
          progress. This is the primary signal, and it does not care which
          direction the sweep runs: an earlier version binary-searched for a
          single NaN boundary assuming finite-at-index-0, which silently degrades
          to nothing if Igor sweeps high-to-low or leaves a permanently NaN tail
          (as these tunes do above ~950 kHz).
        * **the values themselves** -- the fallback for configurations that do not
          NaN-pad at all.

        Sampled values alone are NOT sufficient: most probes land in the un-swept
        NaN region and read constant, so the fingerprint looks stable while the
        sweep is ~10% done. An earlier version did exactly that and called a 30 s
        tune finished after 3 s. Keep the probe count and `tune_stable_polls`
        generous.
        """
        try:
            df = self.igor.DataFolder(r"root:packages:MFP3D:Tune")
            aw, pw = df.Wave("Amp"), df.Wave("Phase")
            n = self._wave_npnts(aw)
            if n < 2:
                return None
            step = max(n // n_probe, 1)
            idx = list(range(0, n, step))
            vals = [aw.GetNumericWavePointValue(i) for i in idx]
            n_finite = sum(1 for v in vals if v == v)          # NaN != NaN
            # a few phase points too, so a phase-only update still registers
            pv = [pw.GetNumericWavePointValue(i) for i in idx[::8]]
            enc = tuple(("nan" if v != v else round(float(v), 12))
                        for v in (vals + pv))
            return (n, n_finite, enc)
        except Exception:
            return None

    def wait_for_tune(self, timeout_s=None, poll_s=None, stable_polls=None,
                      start_timeout_s=15.0, verbose=False):
        """Block until the Igor tune has actually finished.

        `igor.Execute('CanttuneFunc("DoTuneOnceButton")')` only *starts* the
        tune -- Igor runs the sweep as a background task and returns immediately.
        The original code simply slept for a fixed few seconds, which is fine for
        a short tune and silently wrong for a long one: the spectrum gets saved
        mid-sweep, and the caller moves on to change bias or load while the tune
        is still running.

        So poll instead. Two phases, because either alone is unsafe:

        1. **Wait for the waves to CHANGE** (the tune has begun). Without this,
           a wave left over from the previous tune looks stable straight away and
           the function returns instantly.
        2. **Wait for them to STOP changing** for `stable_polls` consecutive
           polls (the sweep has finished filling them).

        Returns True if completion was observed, False on timeout (the caller
        should treat that as suspect data, not as success).
        """
        timeout_s = self.tune_timeout_s if timeout_s is None else timeout_s
        poll_s = self.tune_poll_s if poll_s is None else poll_s
        stable_polls = (self.tune_stable_polls if stable_polls is None
                        else stable_polls)
        t0 = time.time()
        base = self._tune_signature()
        if base is None:
            if verbose:
                print("      cannot read tune waves; falling back to a fixed wait")
            time.sleep(self.tune_settling_time + 2)
            return False

        # phase 1: has the tune started?
        started = False
        while time.time() - t0 < start_timeout_s:
            time.sleep(poll_s)
            sig = self._tune_signature()
            if sig is not None and sig != base:
                started = True
                break
        if not started and verbose:
            print(f"      no change in the tune waves within {start_timeout_s:.0f} s "
                  "— either the tune is very fast or it never started")

        # phase 2: has it stopped changing? Never accept completion before a
        # minimum quiet window has actually elapsed.
        min_quiet_s = poll_s * stable_polls
        prev, stable, last_change = None, 0, time.time()
        while time.time() - t0 < timeout_s:
            time.sleep(poll_s)
            sig = self._tune_signature()
            if sig is None:
                continue
            if prev is not None and sig == prev:
                stable += 1
                if (stable >= stable_polls
                        and time.time() - last_change >= min_quiet_s):
                    if verbose:
                        print(f"      tune finished after {time.time() - t0:.1f} s")
                    time.sleep(self.tune_settling_time)     # let it settle
                    return True
            else:
                stable = 0
                last_change = time.time()
            prev = sig

        print(f"    WARNING: tune did not settle within {timeout_s:.0f} s "
              "(tune_timeout_s). The spectrum may be incomplete — raise "
              "tune_timeout_s, or shorten the tune (fewer points / shorter dwell).")
        return False

    def do_tune(self, wait_time=5, wait_for_complete=True, verbose=False):
        """Start a tune and (by default) wait until it has actually finished.

        `wait_for_complete=False` restores the original fixed-sleep behaviour.
        """
        self.igor.Execute('CanttuneFunc("DoTuneOnceButton")')
        if wait_for_complete:
            return self.wait_for_tune(verbose=verbose)
        time.sleep(wait_time)
        return None

    def get_tune_data(self):
        """Read Frequency / Phase / Amp out of Igor over COM.

        Prefer `read_tune_txt` on the file written by `save_tune` — it is both
        complete and far faster than several thousand COM round-trips. This
        remains as a fallback.
        """
        try:
            self.igor.Execute('SetDataFolder root:packages:MFP3D:Tune')
            df = self.igor.DataFolder(r"root:packages:MFP3D:Tune")
            fw, pw, aw = df.Wave("Frequency"), df.Wave("Phase"), df.Wave("Amp")
            n = self._wave_npnts(fw)
            if n < self.MIN_TUNE_POINTS:
                raise RuntimeError(
                    f"Igor reported only {n} tune points; the tune wave read is "
                    "truncated (see AFMLaserSweepAutomation._wave_npnts).")
            freq = np.array([fw.GetNumericWavePointValue(i) for i in range(n)])
            phase = np.array([pw.GetNumericWavePointValue(i) for i in range(n)])
            amp = np.array([aw.GetNumericWavePointValue(i) for i in range(n)])
            self.igor.Execute('SetDataFolder root:')
            return {'frequency': freq, 'phase': phase, 'amplitude': amp}
        except Exception as e:
            print(f"    WARNING: could not get tune data over COM: {e}")
            self.igor.Execute('SetDataFolder root:')
            return None

    def save_tune(self, position_label="", scan_index=0):
        tune_filename = f"Tune_{self.base_filename}_{position_label}_{scan_index:04d}.txt"
        self.igor.Execute(f'NewPath/O TuneSavePath "{self.windows2igor(self.file_loc)}"')
        self.igor.Execute('SetDataFolder root:packages:MFP3D:Tune')
        self.igor.Execute(f'Save/T/O/P=TuneSavePath Frequency, Phase, Amp as "{tune_filename}"')
        self.igor.Execute('SetDataFolder root:')
        time.sleep(0.5)
        return os.path.join(self.file_loc, tune_filename)

    def extract_resonance_from_tune(self, tune_data, band_Hz=None):
        """Resonance frequency and Q from a tune.

        Peak located by parabolic interpolation on the three points around the
        amplitude maximum, so f_res is not quantised to the frequency step.
        `band_Hz=(lo, hi)` restricts the search, which matters when the tune
        window contains more than one eigenmode. Prints a warning if the peak
        sits on the edge of the search range or if the resonance is spanned by
        too few points to be trustworthy.
        """
        if tune_data is None:
            return np.nan, np.nan
        freq = np.asarray(tune_data['frequency'], float)
        amp = np.asarray(tune_data['amplitude'], float)
        good = np.isfinite(freq) & np.isfinite(amp)
        freq, amp = freq[good], amp[good]
        if freq.size == 0:
            return np.nan, np.nan
        band_Hz = band_Hz if band_Hz is not None else self.resonance_band_Hz
        if band_Hz is not None:
            m = (freq >= band_Hz[0]) & (freq <= band_Hz[1])
            if m.sum() >= 8:
                freq, amp = freq[m], amp[m]
        if freq.size < 3:
            return (freq[int(np.argmax(amp))] if freq.size else np.nan), np.nan

        pk = int(np.argmax(amp))
        if pk in (0, freq.size - 1):
            print(f"    WARNING: tune peak is at the EDGE of the search range "
                  f"({freq[pk] / 1e3:.1f} kHz, range {freq[0] / 1e3:.1f}-"
                  f"{freq[-1] / 1e3:.1f} kHz) — the resonance is probably "
                  f"outside the window set in Igor.")
            f_res = float(freq[pk])
        else:
            y0, y1, y2 = amp[pk - 1], amp[pk], amp[pk + 1]
            den = (y0 - 2 * y1 + y2)
            shift = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            shift = float(np.clip(shift, -1, 1))
            f_res = float(freq[pk] + shift * (freq[1] - freq[0]))

        try:
            above = amp > amp[pk] / np.sqrt(2)
            idx = np.where(above)[0]
            bw = freq[idx[-1]] - freq[idx[0]] if len(idx) > 1 else np.nan
            Q = f_res / bw if bw and bw > 0 else np.nan
            n_fwhm = bw / abs(freq[1] - freq[0]) if bw and bw > 0 else 0
            if n_fwhm and n_fwhm < 5:
                print(f"    WARNING: resonance FWHM spans only ~{n_fwhm:.0f} "
                      f"frequency points ({abs(freq[1] - freq[0]):.0f} Hz step). "
                      "Narrow the tune window or raise the tune point count — "
                      "the line shape is undersampled and the null-spot fit "
                      "will be poor.")
        except Exception:
            Q = np.nan
        return f_res, Q

    #: poll for tune completion instead of sleeping a fixed time. Leave True;
    #: False only reproduces the old (unsafe for a condition series) behaviour.
    wait_for_tune_complete = True
    verbose_tune = False

    def tune_eigenmode(self, position_label="", scan_index=0, save_tune_data=True):
        """Tune, save, and return the FULL complex spectrum.

        Order matters: we save first and read the saved file, because the file
        Igor writes is guaranteed complete whereas the COM read has to be told
        how many points to fetch.
        """
        completed = self.do_tune(wait_time=self.tune_settling_time + 2,
                                 wait_for_complete=self.wait_for_tune_complete,
                                 verbose=self.verbose_tune)

        td, tune_file = None, None
        if save_tune_data:
            try:
                tune_file = self.save_tune(position_label, scan_index)
                td = read_tune_txt(tune_file)
            except Exception as e:
                print(f"    WARNING: could not save/read tune file ({e}); "
                      "falling back to the COM read.")
                td = None
        if td is None or np.size(td['frequency']) < self.MIN_TUNE_POINTS:
            td = self.get_tune_data()

        f_res, Q = self.extract_resonance_from_tune(td)
        out = {'resonance_freq': f_res, 'q_factor': Q, 'tune_data': td,
               'n_points': int(np.size(td['frequency'])) if td else 0,
               'tune_completed': completed}
        if tune_file:
            out['tune_file'] = tune_file
        return out


# --------------------------------------------------------------------------- #
#  Instrument adapter for the online loop                                     #
# --------------------------------------------------------------------------- #
class AsylumInstrument(Instrument):
    """Expose the Asylum automation as measure_at(x_um) for the AL loop.

    Per position: withdraw -> move -> optical image -> (AutoWedge+InvOLS) ->
    engage at load -> tune (fixed window) -> read complex spectrum -> withdraw.

    Coordinates
    -----------
    Positions are absolute micrometers in the LIBRARY's frame, which the rest
    of the package assumes throughout: **x = 0 is the clamped base and x
    increases toward the free end**. `VirtualInstrument.measure_at` documents
    the same convention and `lowrank._signed_zero_crossing` resolves ties to
    "nearest the free end" by taking the largest x, so this frame must not be
    flipped — a D-NS of 111.5 µm on a 120 µm span means 8.5 µm in from the free
    end. Use `um_from_free_end()` to convert.

    Where the laser physically *starts* is a separate question, set by
    `start_at`. The hardware only moves relatively (`DoLDMove`), so this class
    tracks the current position and moves by the delta.

    Parameters
    ----------
    x_start_um : float or None
        Where the spot actually is when this object is created, in library
        coordinates. Overrides `start_at`. Position is tracked in software from
        relative moves, so this MUST match reality -- if you build a second
        instrument after a run that left the laser mid-span, pass
        `x_start_um=first_inst.current_x` (or just reuse the first instrument,
        which is better). Asserting 'free_end' when the spot is elsewhere offsets
        every subsequent move by the difference and will drive it off the end.
    start_at : {'base', 'free_end'}
        Where the detection spot is parked when this object is created. Only used
        when `x_start_um` is None.
        'free_end' requires `span_um` and sets the starting coordinate to
        `span_um`, so every subsequent position is *inward* along the beam.
        With the old default ('base', x_start_um=0) parking at the free end
        made the first move run further OFF the cantilever, because the loop
        thought it was starting at the base and stepped toward the tip.
    span_um : float or None
        Length of the accessible span, i.e. `x_grid[-1]`. Required for
        `start_at='free_end'`, and used for the travel guard.
    hw_sign_toward_free_end : {+1, -1}
        Sign of `DoLDMove` that moves the spot toward the free end. +1 on this
        Cypher (parking at the free end and stepping +x runs off the end).
        Flip it if `preview_moves()` disagrees with what the optical image
        shows. Get this wrong and the loop drives the spot off the cantilever,
        so check it once with `preview_moves()` before engaging.
    x_limits_um : (lo, hi) or None
        Reachable range of x. Defaults to [0, span_um]. On a stiff probe the
        base end is often unreachable, so pass `(x_grid[0], x_grid[-1])` — the
        loop then refuses any position it could not physically reach, instead of
        driving the spot to it.
    travel_margin_um : float
        Slack allowed outside `x_limits_um` before `measure_at` refuses to move.
    analysis_band_Hz : (lo, hi) or None
        Frequency band handed to the reconstruction. The FULL tune is always
        saved to disk; this only crops what the low-rank fit sees. Set it to
        bracket ONE eigenmode's resonance + antiresonance. Leave None only if
        the Igor tune window already contains exactly one mode — the D-NS
        estimator's notch search window is a fraction of the span it is given,
        so handing it a multi-megahertz window makes that search meaningless.
    min_points : int
        Hard floor on the number of frequency points accepted per position.
        Guards against a truncated instrument read ever reaching the fit again.
    """

    def __init__(self, automation, load_nN, dc_bias_V=0.0, x_start_um=None,
                 recalibrate_each=True, position_tag="AM",
                 analysis_band_Hz=None, min_points=32,
                 start_at="base", span_um=None, x_limits_um=None,
                 hw_sign_toward_free_end=+1, travel_margin_um=0.0,
                 apply_bias=True):
        self.a = automation
        self.load_nN = float(load_nN)
        self.dc_bias_V = float(dc_bias_V)
        self.recalibrate_each = recalibrate_each
        self.position_tag = position_tag
        self.analysis_band_Hz = analysis_band_Hz
        self.min_points = int(min_points)

        # -- coordinate frame ------------------------------------------------
        start_at = str(start_at).lower()
        if start_at not in ("base", "free_end"):
            raise ValueError("start_at must be 'base' or 'free_end'")
        if hw_sign_toward_free_end not in (1, -1, +1.0, -1.0):
            raise ValueError("hw_sign_toward_free_end must be +1 or -1")
        self.start_at = start_at
        self.span_um = None if span_um is None else float(span_um)
        self.hw_sign = int(hw_sign_toward_free_end)
        self.travel_margin_um = float(travel_margin_um)
        # Travel guard bounds. Default to the whole beam, but on a stiff probe
        # the base end is often unreachable -- pass x_limits_um=(x_grid[0],
        # x_grid[-1]) so the loop can never ask for a position you cannot reach.
        if x_limits_um is not None:
            self.x_limits_um = (float(min(x_limits_um)), float(max(x_limits_um)))
        elif self.span_um is not None:
            self.x_limits_um = (0.0, self.span_um)
        else:
            self.x_limits_um = None

        if x_start_um is None:
            if start_at == "free_end":
                if self.span_um is None:
                    raise ValueError(
                        "start_at='free_end' needs span_um (= x_grid[-1]) so the "
                        "starting coordinate is known; x=0 is the clamped base "
                        "and x increases toward the free end.")
                x_start_um = self.span_um
            else:
                x_start_um = 0.0
        self.x_start_um = float(x_start_um)
        self.current_x = float(x_start_um)

        if analysis_band_Hz is not None:
            self.a.resonance_band_Hz = tuple(analysis_band_Hz)
        self._invols = None
        self._spring = None
        #: which marked sample spot the scanner is at (None = wherever it was
        #: left; set by goto_spot / measure_conditions_at)
        self.current_spot = None
        self.records = []          # per-position metadata (mirrors the CSV log)
        self._scan_index = 0
        # apply DC bias once
        if apply_bias:
            self.apply_dc_bias()

    def apply_dc_bias(self):
        """(Re-)apply the configured DC bias on Output.A.

        `close()` zeroes the bias, so call this before reusing the same
        instrument for a second acquisition (e.g. the dense reference sweep).
        Reusing one instrument is strongly preferred over building a new one:
        position is tracked in software, so a fresh object has no idea where the
        spot actually is.
        """
        self.a.igor.Execute(f'td_wv("Cypher.Output.A", {self.dc_bias_V})')
        time.sleep(1.0)

    # -- coordinate helpers ------------------------------------------------ #
    def um_from_free_end(self, x_um):
        """Convert a library-frame x (0 = base) to distance from the free end."""
        if self.span_um is None:
            raise ValueError("span_um is unknown; pass it to the constructor")
        return self.span_um - float(x_um)

    def _hw_delta(self, x_target):
        """Relative DoLDMove displacement to get from current_x to x_target."""
        return self.hw_sign * (float(x_target) - self.current_x)

    def _check_travel(self, x_target):
        """Refuse positions outside the reachable range."""
        if self.x_limits_um is None:
            return
        lo = self.x_limits_um[0] - self.travel_margin_um
        hi = self.x_limits_um[1] + self.travel_margin_um
        if not (lo <= float(x_target) <= hi):
            raise ValueError(
                f"requested x = {float(x_target):.1f} um is outside the "
                f"reachable range [{lo:.1f}, {hi:.1f}] um — refusing to move. "
                "x = 0 is the clamped base. Check x_limits_um / span_um / "
                "start_at, and that the position grid matches the span you can "
                "actually reach.")

    def set_position(self, x_um, confirm=False):
        """Declare where the spot actually is, without moving anything.

        Position is tracked in software from relative `DoLDMove` calls, so if the
        laser is moved outside this object's knowledge -- by hand in Igor, or by a
        different AsylumInstrument -- the tracked value goes stale and every
        subsequent move inherits the error. Use this to re-sync after verifying
        the real position optically.
        """
        if not confirm:
            raise ValueError(
                "set_position only adjusts bookkeeping, it does not move the "
                "laser. Verify the spot optically first, then pass confirm=True.")
        old = self.current_x
        self.current_x = float(x_um)
        print(f"  tracked position re-synced: {old:.1f} -> {self.current_x:.1f} um "
              "(no motion commanded)")

    def move_to(self, x_um, capture_image=True, warn_above_um=None):
        """Withdraw and drive the spot to `x_um`, updating the tracked position.

        Measures nothing -- use this to park the laser at a known place before an
        acquisition. Honours `x_limits_um`, so it cannot drive past either end of
        the reachable span.
        """
        x_um = float(x_um)
        self._check_travel(x_um)
        dx = self._hw_delta(x_um)
        if warn_above_um is not None and abs(dx) > warn_above_um:
            print(f"  note: large move of {dx:+.1f} um "
                  f"({self.current_x:.1f} -> {x_um:.1f})")
        self.a.withdraw(); time.sleep(1.5)
        if abs(dx) > 1e-6:
            self.a.move_laser_to_position(dx, 0, relative=True)
            time.sleep(1.5)
        self.current_x = x_um
        note = ""
        if self.span_um is not None:
            note = f" ({self.um_from_free_end(x_um):.1f} um from the free end)"
        print(f"  laser moved to x = {x_um:.1f} um{note}  [DoLDMove({dx:+.1f})]")
        if capture_image:
            try:
                self.a.capture_optical_image(
                    position_label=f"PARK_X{abs(1e3 * x_um):06.0f}")
                print("  optical image captured — check the spot is on the "
                      "cantilever before continuing")
            except Exception as e:
                print(f"  optical image failed: {e}")
        return self.current_x

    def preview_moves(self, x_list):
        """Print the hardware moves a sequence of positions would produce.

        Moves nothing. Run this BEFORE engaging to confirm the direction
        convention: starting at the free end, every DoLDMove should carry the
        spot inward, and the first entry should be a move of ~0 if the loop is
        seeded at the position where the laser is already parked.
        """
        x = self.current_x
        print(f"start: x = {x:.1f} um "
              f"({'free end' if self.start_at == 'free_end' else 'base'}"
              + (f", {self.um_from_free_end(x):.1f} um from the free end)"
                 if self.span_um is not None else ")"))
        print(f"  x = 0 is the clamped base; hw_sign_toward_free_end = "
              f"{self.hw_sign:+d}")
        for xt in x_list:
            d = self.hw_sign * (float(xt) - x)
            toward = "toward free end" if d * self.hw_sign > 0 else \
                     ("toward base" if d != 0 else "no move")
            note = ""
            if self.x_limits_um is not None and not (
                    self.x_limits_um[0] - self.travel_margin_um <= float(xt)
                    <= self.x_limits_um[1] + self.travel_margin_um):
                note = "   <-- UNREACHABLE, would be refused"
            print(f"  x = {float(xt):7.1f} um   DoLDMove({d:+8.1f}, 0)   "
                  f"{toward}{note}")
            x = float(xt)
        return None

    # -- acquisition, split so one laser visit can serve many conditions -- #
    def set_dc_bias(self, volts, settle_s=1.0):
        """Change the DC bias on Output.A without moving or re-engaging."""
        self.dc_bias_V = float(volts)
        self.a.igor.Execute(f'td_wv("Cypher.Output.A", {self.dc_bias_V})')
        time.sleep(settle_s)
        return self.dc_bias_V

    def goto_spot(self, spot_number, withdraw_first=True):
        """Withdraw (by default) and move the SAMPLE to a marked spot.

        Sample position selects e.g. the PPLN domain; it is independent of the
        detection-laser position along the cantilever, which is untouched.
        Does not re-engage — the next set_load(reengage=True) does that.
        """
        spot_number = int(spot_number)
        if withdraw_first:
            self.a.withdraw(); time.sleep(1.0)
        ok = self.a.goto_spot(spot_number, wait=True,
                              verbose=getattr(self.a, 'verbose_tune', False))
        self.current_spot = spot_number
        return ok

    def set_load(self, load_nN, reengage=True, settle_s=0.5):
        """Change the applied load via the deflection setpoint.

        `reengage=True` (default) lifts and re-engages, which is the safe way to
        make a large load change; set False only for small steps where you have
        confirmed the feedback settles cleanly, since a big setpoint jump while
        in contact drives the tip hard into the surface.
        """
        if self._invols is None or self._spring is None:
            raise RuntimeError("calibrate first (InvOLS / spring constant unknown)")
        self.load_nN = float(load_nN)
        setpoint_V = (self.load_nN * 1e-9) / self._spring / self._invols
        self.a.igor.Execute(f'PV("DeflectionSetpointVolts", {setpoint_V})')
        time.sleep(settle_s)
        if reengage:
            self.a.simple_engage(wait_time=5); time.sleep(2)
        return setpoint_V

    def _goto_and_prepare(self, x_um, capture_image=True):
        """Withdraw, move to x, optional image, AutoWedge + InvOLS. No engage."""
        a = self.a
        x_um = float(x_um)
        self._check_travel(x_um)
        label = f"X{'m' if x_um < 0 else ''}{abs(1e3 * x_um):06.0f}"
        a.withdraw(); time.sleep(1.5)
        dx = self._hw_delta(x_um)
        if abs(dx) > 1e-6:
            a.move_laser_to_position(dx, 0, relative=True)
            time.sleep(1.5)
        self.current_x = x_um
        if capture_image:
            try:
                a.capture_optical_image(position_label=label)
            except Exception as e:
                print(f"  optical image failed: {e}")
        if self.recalibrate_each or self._invols is None:
            if getattr(a, 'use_autowedge', True):
                a.do_autowedge()
            self._invols = a.measure_invols()
            self._spring = a.get_gmv()['SpringConstant']
        return label

    def _tune_and_read(self, label, setpoint_V):
        """Tune at the current position/condition and return (freq, Z, record)."""
        a = self.a
        bias = f"{'m' if self.dc_bias_V < 0 else 'p'}{abs(self.dc_bias_V):.3f}V".replace('.', 'p')
        spot_tag = f"_S{self.current_spot}" if self.current_spot is not None else ""
        tune = a.tune_eigenmode(
            position_label=f"{label}_DC{bias}_L{self.load_nN:04.0f}nN{spot_tag}",
            scan_index=self._scan_index, save_tune_data=True)
        self._scan_index += 1
        td = tune['tune_data']
        if td is None:
            a.withdraw()
            raise RuntimeError("tune returned no data")
        freq_full = np.asarray(td['frequency'], float)
        if freq_full.size < self.min_points:
            a.withdraw()
            raise RuntimeError(
                f"tune returned only {freq_full.size} frequency points "
                f"(expected the full Igor tune window, >= {self.min_points}). "
                "Refusing to feed a truncated spectrum to the reconstruction — "
                "this is what a bad instrument read looks like.")
        freq, Z = tune_to_complex(td, band_Hz=self.analysis_band_Hz)
        rec = dict(position_x_um=self.current_x,
                   um_from_free_end=(self.span_um - self.current_x
                                     if self.span_um is not None else np.nan),
                   dc_bias_V=self.dc_bias_V,
                   load_nN=self.load_nN, setpoint_V=setpoint_V,
                   sample_spot=self.current_spot,
                   invols_m_per_V=self._invols, spring_N_per_m=self._spring,
                   resonance_freq_Hz=tune['resonance_freq'],
                   q_factor=tune['q_factor'],
                   n_tune_points=int(freq_full.size),
                   tune_f_lo_Hz=float(freq_full[0]), tune_f_hi_Hz=float(freq_full[-1]),
                   n_fit_points=int(freq.size),
                   tune_completed=bool(tune.get('tune_completed', True)),
                   tune_file=tune.get('tune_file'),
                   timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
        self.records.append(rec)
        return freq, Z, rec

    def measure_at(self, x_um):
        """One spectrum at x, at the instrument's current bias and load."""
        label = self._goto_and_prepare(x_um)
        setpoint_V = self.set_load(self.load_nN, reengage=True)
        freq, Z, rec = self._tune_and_read(label, setpoint_V)
        self.a.withdraw(); time.sleep(1)
        return freq, Z, dict(rec, tune_center=self.a.eigenmode_center_freq)

    def measure_conditions_at(self, x_um, conditions, reengage_on_load_change=True,
                              verbose=True):
        """Measure every (bias, load) condition at ONE laser position.

        This is the whole point of a series: the laser move, AutoWedge, InvOLS and
        engage cost roughly a minute, while changing the bias costs about a second
        and changing the load a few. Visiting each position once and sweeping the
        conditions there is therefore many times faster than looping the whole
        acquisition per condition -- and it removes position-repeatability error
        from the comparison between conditions, which matters because the
        bias-dependence fit is done per position.

        `conditions` is a sequence of objects with `.bias_V` and `.load_nN`.
        Returns ``{index: (freq, Z, record)}``; a condition that fails is recorded
        as None rather than aborting the position.
        """
        label = self._goto_and_prepare(x_um)
        out, last_load = {}, None
        for i, cond in enumerate(conditions):
            try:
                # sample-spot change first: it withdraws, so bias/load/engage
                # must come after. A condition with spot=None never moves the
                # sample, so plain bias/load series behave exactly as before.
                spot = getattr(cond, 'spot', None)
                spot_changed = (spot is not None and spot != self.current_spot)
                if spot_changed:
                    self.goto_spot(spot)
                    last_load = None            # force a re-engage at the new spot
                self.set_dc_bias(cond.bias_V)
                need = (last_load is None
                        or abs(cond.load_nN - last_load) > 1e-9)
                setpoint_V = self.set_load(
                    cond.load_nN,
                    reengage=(need and reengage_on_load_change) or last_load is None)
                last_load = cond.load_nN
                freq, Z, rec = self._tune_and_read(label, setpoint_V)
                out[i] = (freq, Z, rec)
                if verbose:
                    flag = "" if rec.get('tune_completed', True) else \
                           "   ** TUNE DID NOT COMPLETE **"
                    print(f"      cond {i}: {cond.bias_V:+.2f} V, "
                          f"{cond.load_nN:.0f} nN -> {freq.size} pts, "
                          f"f_res {rec['resonance_freq_Hz'] / 1e3:.2f} kHz{flag}")
            except Exception as e:
                out[i] = None
                print(f"      cond {i} ({cond.bias_V:+.2f} V, {cond.load_nN:.0f} nN) "
                      f"FAILED: {type(e).__name__}: {e}")
        self.a.withdraw(); time.sleep(1)
        return out

    def save_log(self, path=None):
        import pandas as pd
        path = path or os.path.join(self.a.file_loc, "ActiveModeMap_log.csv")
        pd.DataFrame(self.records).to_csv(path, index=False)
        return path

    def close(self):
        self.a.withdraw()
        self.a.igor.Execute('td_wv("Cypher.Output.A", 0)')
