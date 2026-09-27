# --- sparse capture with a FULL-RESPONSE uncertainty target -------------------------
# Drives LowRankModeMap directly (the class docstring's own loop) instead of run_series,
# whose stop rule is D-NS-only. Positions are chosen D-optimally on the SAME 1 um grid the
# dense map used, so the comparison is position-matched. Stop when, for STABLE_OVER
# consecutive additions, both hold in every resonance band:
#   (a) median predictive sigma / |Z_rec| over the grid  <= TARGET_REL
#   (b) the map stopped moving: NRMSE(Z_rec_n, Z_rec_{n-1}) <= TARGET_REL
# Never before RANK + 2 positions (honest residual dof), never past MAX_POS.
import numpy as np, os, time, json
from activemodemap.lowrank import LowRankModeMap, band_mask
from activemodemap.series import make_conditions, save_checkpoint

def band_stats(rec, freq, bands):
    Zr, sd = rec["Zrec"], rec["std"]
    out = []
    for lo, hi in bands:
        m = band_mask(freq, (lo, hi))
        rel = sd[:, m] / np.maximum(np.abs(Zr[:, m]), 1e-12)
        out.append(float(np.median(rel)))
    return out

def band_change(rec, prev, freq, bands):
    if prev is None:
        return [np.inf] * len(bands)
    out = []
    for lo, hi in bands:
        m = band_mask(freq, (lo, hi))
        a, b = rec["Zrec"][:, m], prev["Zrec"][:, m]
        out.append(float(np.linalg.norm(a - b) / max(np.linalg.norm(a), 1e-12)))
    return out

def run_sparse_capture(inst, x_grid_um, bands_Hz, *, rank=6, target_rel=0.10, stable_over=3,
                       max_pos=30, start_near_um=None, dns_band_Hz=None, checkpoint_path=None,
                       load_nN=None, bias_V=None, required=None, verbose=True):
    # `required`: indices into bands_Hz that must meet the target; the rest are reported only
    required = list(range(len(bands_Hz))) if required is None else list(required)
    x_grid_um = np.asarray(x_grid_um, float)
    cond = make_conditions([bias_V if bias_V is not None else inst.dc_bias_V],
                           [load_nN if load_nN is not None else inst.load_nN], vary="bias_inner")
    mm = LowRankModeMap(x_grid_um, rank=rank, start_near_um=start_near_um, dns_band_Hz=dns_band_Hz)
    measured, prev, ok_streak, failures, blocked = {}, None, 0, 0, set()
    t0 = time.time()
    if verbose:
        print(f"sparse capture: rank {rank}, min {mm.min_positions}, max {max_pos}, target {target_rel:.0%} "
              f"in bands {required} of {len(bands_Hz)}, stable over {stable_over}; grid {x_grid_um[0]:.0f}-{x_grid_um[-1]:.0f} um "
              f"({x_grid_um.size} candidates); condition {cond[0]}")
    while mm.n < max_pos:
        x = mm.next_position()
        if verbose:
            print(f"\n=== sparse position {mm.n + 1}: x = {x:.1f} um ===", flush=True)
        try:
            got = inst.measure_conditions_at(x, cond, verbose=verbose)
        except KeyboardInterrupt:
            print("interrupted -- checkpoint holds everything up to here"); break
        except Exception as e:
            print(f"position {x:.1f} failed ({type(e).__name__}: {e}); stopping"); break
        if got.get(0) is None:
            mm.block_position(x); blocked.add(float(x)); failures += 1
            print(f"  failed at x={x:.1f}; blocked ({failures} consecutive)")
            if failures >= 3:
                print("  THREE IN A ROW - stopping"); break
            continue
        failures = 0
        f, Z = got[0][0], got[0][1]
        measured[float(x)] = {0: (f, Z)}
        mm.add_measurement(float(x), f, Z)
        if checkpoint_path:
            save_checkpoint(checkpoint_path, measured, cond, 0)
        if mm.n < mm.min_positions:
            if verbose:
                print(f"  {mm.n}/{mm.min_positions} positions before the first honest reconstruction "
                      f"({(time.time()-t0)/60:.1f} min)")
            continue
        rec = mm.reconstruct(nboot=100)
        rel = band_stats(rec, mm.freq, bands_Hz)
        chg = band_change(rec, prev, mm.freq, bands_Hz)
        prev = rec
        hit = all(rel[i] <= target_rel for i in required) and all(chg[i] <= target_rel for i in required)
        ok_streak = ok_streak + 1 if hit else 0
        if verbose:
            print(f"  n={mm.n}  sigma/|Z| per band: " + " ".join(f"{r:.3f}" for r in rel)
                  + "   map change: " + " ".join(f"{c:.3f}" if np.isfinite(c) else "  -  " for c in chg)
                  + f"   D-NS {rec['dns']:.2f} +/- {rec['dns_ci']/2:.2f} um   "
                  + f"{'OK' if hit else '..'} streak {ok_streak}/{stable_over}   ({(time.time()-t0)/60:.1f} min)")
        if ok_streak >= stable_over:
            if verbose:
                print(f"\nconverged: target {target_rel:.0%} met in every band for {stable_over} consecutive additions")
            break
    else:
        if verbose:
            print(f"\nstopped at the cap of {max_pos} positions without meeting the target")
    if checkpoint_path:
        save_checkpoint(checkpoint_path, measured, cond, 0)
    try:
        inst.a.withdraw()
    except Exception as e:
        print("withdraw:", e)
    if verbose and mm.n >= mm.min_positions:
        print("\nrank sensitivity of the final map (the honest uncertainty):")
        try:
            mm.rank_sensitivity()
        except Exception as e:
            print("  rank_sensitivity:", e)
    return dict(mm=mm, measured=measured, x_sel=sorted(measured), n=mm.n, rec=prev, blocked=blocked)
