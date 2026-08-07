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

IMPORTANT — fixed tune window. For mode-shape mapping the tune must cover the
whole resonance + antiresonance structure and must NOT auto-recenter per
position. Set a fixed, wide tune range in Igor before running the loop (the same
way you set it for the dense sweep); `measure_at` reads whatever window Igor is
configured for and never changes it.
"""

from __future__ import annotations

import os
import re
import time
import numpy as np

from .online import Instrument


# --------------------------------------------------------------------------- #
#  Automation class (adapted from afm_laser_sweep_automation_v4)              #
# --------------------------------------------------------------------------- #
class AFMLaserSweepAutomation:
    """Automated laser positioning, calibration, engage, and tune on Asylum/Igor."""

    def __init__(self, igor, file_loc, base_filename, log_filename):
        self.igor = igor
        self.file_loc = file_loc
        self.base_filename = base_filename
        self.log_filename = log_filename
        self.load_force_setpoint = 0.5
        self.tune_settling_time = 2.0
        self.invols_bounds = (4e-8, 10e-7)
        self.use_autowedge = False
        self.autowedge_pause = 15.0
        self.eigenmode_center_freq = 70000
        self.results = []

    # -- low-level ------------------------------------------------------- #
    def ex(self, variable="", panel="", val=0, string="", verbose=False):
        line = ("print " if verbose else "") + \
            f'ARExecuteControl("{variable}", "{panel}", {val}, "{string}")'
        self.igor.Execute(line)
        return line

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
            curr = float(str(file['wave']['note']).split('rInvOLS: ')[1].split('\\')[0])
            deflV = defl / curr
            di = max_i - engage_i
            i1 = int(engage_i + 0.25 * di); i2 = int(max_i - 0.25 * di)
            invols = (z[i2] - z[i1]) / (deflV[i2] - deflV[i1])
            if self.invols_bounds[0] < invols < self.invols_bounds[1]:
                self.ex('InvOLSSetVar_1', 'MasterPanel', invols)
            return invols
        except Exception as e:
            print(f"  ERROR loading force curve: {e}")
            return None

    # -- tune ------------------------------------------------------------ #
    def do_tune(self, wait_time=5):
        self.igor.Execute('CanttuneFunc("DoTuneOnceButton")')
        time.sleep(wait_time)

    def get_tune_data(self):
        try:
            self.igor.Execute('SetDataFolder root:packages:MFP3D:Tune')
            fw = self.igor.DataFolder(r"root:packages:MFP3D:Tune").Wave("Frequency")
            pw = self.igor.DataFolder(r"root:packages:MFP3D:Tune").Wave("Phase")
            aw = self.igor.DataFolder(r"root:packages:MFP3D:Tune").Wave("Amp")
            n = fw.GetDimensions()[0]
            freq = np.array([fw.GetNumericWavePointValue(i) for i in range(n)])
            phase = np.array([pw.GetNumericWavePointValue(i) for i in range(n)])
            amp = np.array([aw.GetNumericWavePointValue(i) for i in range(n)])
            self.igor.Execute('SetDataFolder root:')
            return {'frequency': freq, 'phase': phase, 'amplitude': amp}
        except Exception as e:
            print(f"    WARNING: could not get tune data: {e}")
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

    def extract_resonance_from_tune(self, tune_data):
        if tune_data is None:
            return np.nan, np.nan
        freq, amp = tune_data['frequency'], tune_data['amplitude']
        pk = np.argmax(amp); f_res = freq[pk]
        try:
            above = amp > amp[pk] / np.sqrt(2)
            idx = np.where(above)[0]
            bw = freq[idx[-1]] - freq[idx[0]] if len(idx) > 1 else np.nan
            Q = f_res / bw if bw and bw > 0 else np.nan
        except Exception:
            Q = np.nan
        return f_res, Q

    def tune_eigenmode(self, position_label="", scan_index=0, save_tune_data=True):
        self.do_tune(wait_time=self.tune_settling_time + 2)
        td = self.get_tune_data()
        f_res, Q = self.extract_resonance_from_tune(td)
        out = {'resonance_freq': f_res, 'q_factor': Q, 'tune_data': td}
        if save_tune_data:
            try:
                out['tune_file'] = self.save_tune(position_label, scan_index)
            except Exception as e:
                print(f"    WARNING: could not save tune: {e}")
        return out


# --------------------------------------------------------------------------- #
#  Instrument adapter for the online loop                                     #
# --------------------------------------------------------------------------- #
class AsylumInstrument(Instrument):
    """Expose the Asylum automation as measure_at(x_um) for the AL loop.

    Positions are ABSOLUTE micrometers measured from the start position where
    the laser sits when this object is created (x = x_start_um). Because the
    hardware moves relatively, we track the current position and move by the
    delta; the tip is always withdrawn before the laser motor moves.

    Per position: withdraw -> move -> optical image -> (AutoWedge+InvOLS) ->
    engage at load -> tune (fixed window) -> read complex spectrum -> withdraw.
    """

    def __init__(self, automation, load_nN, dc_bias_V=0.0, x_start_um=0.0,
                 recalibrate_each=True, position_tag="AM"):
        self.a = automation
        self.load_nN = float(load_nN)
        self.dc_bias_V = float(dc_bias_V)
        self.current_x = float(x_start_um)
        self.recalibrate_each = recalibrate_each
        self.position_tag = position_tag
        self._invols = None
        self._spring = None
        self.records = []          # per-position metadata (mirrors the CSV log)
        self._scan_index = 0
        # apply DC bias once
        self.a.igor.Execute(f'td_wv("Cypher.Output.A", {self.dc_bias_V})')
        time.sleep(1.0)

    def measure_at(self, x_um):
        a = self.a
        x_um = float(x_um)
        label = f"X{1e3 * x_um:06.0f}"

        # 1. withdraw, then move laser by the delta to the requested absolute x
        a.withdraw(); time.sleep(1.5)
        dx = x_um - self.current_x
        if abs(dx) > 1e-6:
            a.move_laser_to_position(dx, 0, relative=True)
            time.sleep(1.5)
        self.current_x = x_um

        # 2. optical image (spot verification)
        try:
            a.capture_optical_image(position_label=label)
        except Exception as e:
            print(f"  ⚠ optical image failed: {e}")

        # 3. calibration (AutoWedge + InvOLS); optionally reuse first value
        if self.recalibrate_each or self._invols is None:
            a.do_autowedge()
            self._invols = a.measure_invols()
            self._spring = a.get_gmv()['SpringConstant']
        invols, spring = self._invols, self._spring

        # 4. engage at the requested load
        setpoint_V = (self.load_nN * 1e-9) / spring / invols
        a.igor.Execute(f'PV("DeflectionSetpointVolts", {setpoint_V})')
        time.sleep(0.5)
        a.simple_engage(wait_time=5); time.sleep(2)

        # 5. tune (fixed window, set in Igor) and read the complex spectrum
        bias = f"{'m' if self.dc_bias_V < 0 else 'p'}{abs(self.dc_bias_V):.3f}V".replace('.', 'p')
        tune = a.tune_eigenmode(
            position_label=f"{label}_DC{bias}_L{self.load_nN:04.0f}nN",
            scan_index=self._scan_index, save_tune_data=True)
        self._scan_index += 1
        td = tune['tune_data']
        if td is None:
            a.withdraw()
            raise RuntimeError("tune returned no data")
        freq = np.asarray(td['frequency'], float)
        Z = np.asarray(td['amplitude'], float) * np.exp(
            1j * np.deg2rad(np.asarray(td['phase'], float)))

        # 6. withdraw and record
        a.withdraw(); time.sleep(1)
        rec = dict(position_x_um=x_um, dc_bias_V=self.dc_bias_V,
                   load_nN=self.load_nN, setpoint_V=setpoint_V,
                   invols_m_per_V=invols, resonance_freq_Hz=tune['resonance_freq'],
                   q_factor=tune['q_factor'], tune_file=tune.get('tune_file'),
                   timestamp=time.strftime('%Y-%m-%d %H:%M:%S'))
        self.records.append(rec)
        meta = dict(rec, tune_center=self.a.eigenmode_center_freq)
        return freq, Z, meta

    def save_log(self, path=None):
        import pandas as pd
        path = path or os.path.join(self.a.file_loc, "ActiveModeMap_log.csv")
        pd.DataFrame(self.records).to_csv(path, index=False)
        return path

    def close(self):
        self.a.withdraw()
        self.a.igor.Execute('td_wv("Cypher.Output.A", 0)')
