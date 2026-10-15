"""Template for the BE acquisition adapter used by stages.run_imaging().

Copy this file to `smm_be_adapter.py` next to the notebook (or anywhere on sys.path) and fill in
`be_frame()` with the call that acquires ONE band-excitation frame on this instrument and returns
the path of the file it wrote. `cfg.imaging.be_acquire = "smm_be_adapter:be_frame"` points at it.

The stages take care of everything around the call: laser at the stop, InvOLS, engage at the
capture load, tune (CR1 and FWHM -> band centre and width), DC bias set and verified, bias back to
0 V afterwards. The adapter must NOT move the laser, change the setpoint, or withdraw.

Known hazards from the hafnia bring-up (2026-10-05), to keep guarded inside your BE code:
  * BEPyAE init writes DeflectionSetpointVolts = -10 V -> read the setpoint before, restore after with
    MainSetpointFunc(v, "", "DeflectionSetpointVolts").
  * Trigger must leave on the ARC crosspoint BNCOut1 = OutB; SignalTap on QPDI AC displacement.
  * Use the +/-2 V analog-input range; centre the band on a trusted CR1, not on a weak corner re-tune.
"""


def be_frame(inst, name, v_dc, v_ac, f_center_Hz, f_width_Hz, out_dir, px=128, repeats=1, **extra):
    """Acquire one BE frame and return the path of the written file (.hf5 or .npz).

    Parameters are what the stages know: `inst` (the AsylumInstrument; `inst.a.igor` is the COM
    object), a frame `name` for the file, the DC bias already applied, V_ac, the band centre and
    width in Hz, the output folder, pixels per side and repeats. `extra` carries the full
    ImagingConfig as a dict for anything else.
    """
    # Example shape, adapt to your module (quant_be_generic / BEPyAE wrapper):
    #
    #   from quant_be_generic import be_frame as _be
    #   return _be(igor=inst.a.igor, f_center=f_center_Hz, f_width=f_width_Hz, v_ac=v_ac, v_dc=v_dc,
    #              px=px, repeats=repeats, out_dir=out_dir, base=f"BE_{name}")
    #
    raise NotImplementedError("fill in be_frame() with this instrument's BE acquisition call")
