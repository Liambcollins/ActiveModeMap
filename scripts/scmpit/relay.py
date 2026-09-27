r"""File relay: lets scripts dropped into a queue folder run inside THIS kernel.

Why: the kernel that holds `inst` (Igor COM, main-thread only) cannot be reached
from the Claude session, but the data folder can. So this cell watches
    D:\User Data\Liam\ActiveModeMap\DomainsB_SCMPIT\_relay\queue\
and executes each *.py it finds, in filename order, in the notebook namespace --
exactly what `%run -i <script>` would do -- then moves it to done\ or failed\.

RUN ONCE, FROM THE KERNEL THAT HOLDS (OR WILL HOLD) `inst`, ON THE MAIN THREAD:

    %run -i "C:/Users/Asylum User/Desktop/STAFF Software/Liam/ActiveModeMap/scripts/scmpit/relay.py"

The cell blocks while it runs (that is the point: the scripts need the main
thread). To stop it: Kernel -> Interrupt, or create the file _relay\STOP.
Interrupting mid-script triggers that script's own `finally` (bias 0 V, withdraw).

Status: _relay\status.json (heartbeat every poll), _relay\relay_log.txt.
Every executed script also keeps its own log next to its checkpoint.
"""
import glob
import io
import json
import os
import shutil
import sys
import time
import traceback

RELAY_ROOT = r'D:\User Data\Liam\ActiveModeMap\DomainsB_SCMPIT\_relay'
Q_DIR, RUN_DIR = os.path.join(RELAY_ROOT, 'queue'), os.path.join(RELAY_ROOT, 'running')
DONE_DIR, FAIL_DIR = os.path.join(RELAY_ROOT, 'done'), os.path.join(RELAY_ROOT, 'failed')
STOP_FILE = os.path.join(RELAY_ROOT, 'STOP')
STATUS = os.path.join(RELAY_ROOT, 'status.json')
RLOG = os.path.join(RELAY_ROOT, 'relay_log.txt')
POLL_S = 5.0
# queued scripts are COPIES; their shared module lives with the canonical ones here
SCRIPTS_DIR = r'C:\Users\Asylum User\Desktop\STAFF Software\Liam\ActiveModeMap\scripts\scmpit'
REPO_DIR = r'C:\Users\Asylum User\Desktop\STAFF Software\Liam\ActiveModeMap'
for _p in (SCRIPTS_DIR, REPO_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

for _d in (Q_DIR, RUN_DIR, DONE_DIR, FAIL_DIR):
    os.makedirs(_d, exist_ok=True)


def _rlog(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(RLOG, 'a', encoding='utf-8') as f:
        f.write(line + '\n')


def _status(state, current=None, extra=None):
    d = dict(state=state, current=current, time=time.strftime('%Y-%m-%d %H:%M:%S'),
             queued=sorted(os.path.basename(p) for p in glob.glob(os.path.join(Q_DIR, '*.py'))),
             has_inst=('inst' in globals()),
             inst_x=(getattr(globals().get('inst'), 'current_x', None)),
             inst_spot=(getattr(globals().get('inst'), 'current_spot', None)),
             inst_load=(getattr(globals().get('inst'), 'load_nN', None)),
             inst_bias=(getattr(globals().get('inst'), 'dc_bias_V', None)))
    if extra:
        d.update(extra)
    tmp = STATUS + '.tmp'
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(d, f, indent=1, default=str)
    os.replace(tmp, STATUS)


def _safe_park():
    """Best-effort: bias to 0 V and withdraw, if an instrument exists."""
    inst_ = globals().get('inst')
    if inst_ is None:
        return
    try:
        inst_.set_dc_bias(0.0)
    except Exception as e:
        _rlog(f'  (bias reset failed: {e})')
    try:
        inst_.a.withdraw()
    except Exception as e:
        _rlog(f'  (withdraw failed: {e})')


def _run_one(path):
    name = os.path.basename(path)
    run_path = os.path.join(RUN_DIR, name)
    shutil.move(path, run_path)
    _rlog(f'>>> START {name}')
    _status('running', name)
    t0 = time.time()
    ok = True
    try:
        with open(run_path, 'r', encoding='utf-8') as f:
            src = f.read()
        code = compile(src, run_path, 'exec')
        g = globals()
        g['__file__'] = run_path
        exec(code, g)                      # == %run -i, in this namespace
    except KeyboardInterrupt:
        ok = False
        _rlog(f'!!! INTERRUPTED during {name}')
        raise
    except SystemExit as e:
        ok = (e.code in (None, 0))
        _rlog(f'    {name} exited with SystemExit({e.code})')
    except Exception:
        ok = False
        tb = traceback.format_exc()
        _rlog(f'!!! FAILED {name}\n{tb}')
        with open(os.path.join(FAIL_DIR, name + '.traceback.txt'), 'w', encoding='utf-8') as f:
            f.write(tb)
    finally:
        mins = (time.time() - t0) / 60.0
        dest = os.path.join(DONE_DIR if ok else FAIL_DIR,
                            f"{time.strftime('%H%M%S')}_{name}")
        try:
            shutil.move(run_path, dest)
        except Exception as e:
            _rlog(f'  (could not move {name}: {e})')
        _rlog(f'<<< {"DONE" if ok else "FAILED"} {name}  ({mins:.1f} min)')
    return ok


_rlog('=' * 70)
_rlog(f'relay started; watching {Q_DIR}  (poll {POLL_S:.0f} s; STOP file or Kernel->Interrupt to end)')
_rlog(f"'inst' in namespace: {'inst' in globals()}")
try:
    while True:
        if os.path.exists(STOP_FILE):
            _rlog('STOP file found -- leaving relay loop (STOP file removed)')
            try:
                os.remove(STOP_FILE)
            except Exception:
                pass
            break
        queue = sorted(glob.glob(os.path.join(Q_DIR, '*.py')))
        if queue:
            _run_one(queue[0])
            _status('idle')
        else:
            _status('idle')
            time.sleep(POLL_S)
except KeyboardInterrupt:
    _rlog('KeyboardInterrupt -- parking instrument and leaving relay loop')
    _safe_park()
finally:
    _status('stopped')
    _rlog('relay stopped')
