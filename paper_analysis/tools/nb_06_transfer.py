from nbbuild import SETUP

CELLS = [
("md", """
# 06 · The enhancement is not Q: d33 from an on-resonance amplitude (paper Fig. 6)

SCM-PIT-B on PPLN DomainsB, two-domain bias survey (8 positions × 7 biases × 2 domains,
500 nN, V_ac = 1 V), Run 2 (primary) and Run 1 (repeat, before/after a tip-altering load ladder).

For each laser position:
- **QS channel**: complex mean over 15–45 kHz → two-domain fit → |P_QS|, V_cpd, |b|/|P|
- **CR1 channel**: complex value at the CR1 peak → two-domain fit → |P_CR1|
- **E(x) = |P_CR1| / |P_QS|**: the in-situ enhancement; Q from the −3 dB width
- **d33_QS = |P_QS| · InvOLS(x) / 32 / V_ac**: model-free, needs no resonance model
"""),
("code", SETUP),
("code", """
VAC = 1.0
runs = {}
for tag in ("r2", "r1"):
    s = F.load(f"scmpitB_{tag}_bias")
    inv = axis.invols_interpolator(F.load(f"scmpitB_{tag}_preflight"))
    runs[tag] = dict(series=s, invols=inv, table=domains.transfer_table(s, inv, VAC))
    print(tag.upper(), s)
t2, t1 = runs['r2']['table'], runs['r1']['table']
show = lambda t: (t.assign(invols_um_per_V=t.invols * 1e6, P_qs_mV=t.P_qs * 1e3, P_cr_mV=t.P_cr * 1e3)
                   .drop(columns=['invols', 'P_qs', 'P_cr']).round(3))
show(t2)
"""),
("md", """
## Positions in the lever frame
The laser spot drifts along the lever by ~1 µm/h plus jumps at load excursions (notebook 05a).
Positions below are therefore measured from the clamp x₀ fitted to the bias-survey force curves
of each run (static-shape ruler), not from the fixed 6.1 µm stage offset. For R2 the fixed offset
overstates every distance by ~18 µm.
"""),
("code", """
import datetime as dt
from fmmpaper import calib
D0 = {"r1": ("DomainsB_SCMPIT", dt.datetime(2026, 9, 20, 11)),
      "r2": ("DomainsB_SCMPIT_R2", dt.datetime(2026, 9, 20, 22))}
x0_survey = {}
for tag, (camp, d0) in D0.items():
    a = calib.frame_anchors(camp, calib.timeline(camp, d0), 226.5)
    x0_survey[tag] = float(a.set_index("stage").loc["10_bias_survey", "x0"])
    runs[tag]['table']['x_lever_um'] = runs[tag]['table'].x_um - x0_survey[tag]
print({k: round(v, 2) for k, v in x0_survey.items()}, "um (clamp position in stage frame at the survey)")
t2[['x_um', 'x_clamp_um', 'x_lever_um']].round(1).T
"""),
("md", "## Check against the archived R2 reduction (`s3_transfer.json`)"),
("code", """
ref = pd.DataFrame(F.load("scmpitB_r2_s3")['rows'])
cmp = pd.DataFrame(dict(x_clamp=t2.x_clamp_um, E=t2.E, E_ref=ref.E,
                        d33=t2.d33_qs, d33_ref=ref.d33_qs_pm_per_V,
                        vcpd=t2.vcpd_qs, vcpd_ref=ref.vcpd_qs))
print("max |dE|/E:", float(np.max(np.abs(cmp.E / cmp.E_ref - 1))),
      " max |d d33|:", float(np.max(np.abs(cmp.d33 - cmp.d33_ref))), "pm/V")
cmp.round(3)
"""),
("md", """
## E(x) vs Q along the lever, and the blind Euler–Bernoulli prediction
The EB prediction is read from `r2_eb_fit.npz` (the blind fit saw only CR1–3 frequencies,
the static shape of 1/InvOLS and Q; no amplitudes). Porting that fit (`fit_eb_r2.py`) into the
package is a Phase-2 task; the numbers here are the archived ones.
"""),
("code", """
eb = F.load("scmpitB_r2_eb_fit")
assert np.allclose(eb['X'], t2.x_um), "EB fit positions differ from the survey"
interior = t2.x_um < t2.x_um.max()                       # free end excluded, as in the manuscript
ratio = (eb['E_meas'] / eb['E_mod'])[interior]
print(f"blind EB: E_meas/E_mod over interior positions {ratio.min():.2f}-{ratio.max():.2f} "
      f"(sd {100*ratio.std()/ratio.mean():.0f} %); fitted L = {float(eb['L']):.1f} um")
print(f"measured E range {t2.E.min():.1f}-{t2.E.max():.1f}; Q = {t2.Q.mean():.0f} +- {t2.Q.std():.0f}")

fig, axs = fp.figure("double", aspect=0.34, ncols=3)
ax = axs[0]
ax.semilogy(t2.x_lever_um, t2.E, 'o-', color=fp.C['meas'], ms=3.5, label='measured E(x), R2')
ax.semilogy(t1.x_lever_um, t1.E, 'o:', color='0.55', ms=3, label='measured E(x), R1')
ax.semilogy(t2.x_lever_um, eb['E_mod'], 's--', color=fp.C['ebgp'], ms=3.5, label='blind EB prediction')
ax.semilogy(t2.x_lever_um, t2.Q, '-', color=fp.C['gp'], label='Q')
ax.set_xlabel("x − x₀, lever frame (µm)"); ax.set_ylabel("CR1 enhancement"); ax.legend(fontsize=6)
fp.panel_label(ax, "a")

ax = axs[1]
d_eb = domains.d33_from_enhancement(t2, eb['E_mod'], VAC)
ax.plot(t2.x_lever_um, t2.d33_qs, 'o-', color=fp.C['meas'], ms=3.5, label='quasi-static')
ax.plot(t2.x_lever_um, d_eb, 's-', color=fp.C['ebgp'], ms=3.5, label='CR1 ÷ EB-predicted E')
ax.plot(t2.x_lever_um, t2.d33_cr_over_Q, '^--', color=fp.C['gp'], ms=3.5, label='CR1 ÷ Q')
ax.set_yscale('log'); ax.set_xlabel("x − x₀, lever frame (µm)"); ax.set_ylabel("d33 (pm/V)")
ax.legend(fontsize=6); fp.panel_label(ax, "b")

ax = axs[2]
ax.plot(t1.x_lever_um, t1.d33_qs, 'o:', color='0.55', ms=3, label='R1')
ax.plot(t2.x_lever_um, t2.d33_qs, 'o-', color=fp.C['meas'], ms=3.5, label='R2')
ax.set_xlabel("x − x₀, lever frame (µm)"); ax.set_ylabel("quasi-static d33 (pm/V)")
ax.set_ylim(0, 12); ax.legend(fontsize=6); fp.panel_label(ax, "c")
fig.tight_layout(); fp.save(fig, "06_E_vs_Q_d33")

summary = pd.DataFrame({
    "route": ["quasi-static, per-position InvOLS", "CR1 / EB-predicted E (per-position InvOLS)",
              "CR1 / Q (the usual shortcut)"],
    "median_pm_per_V": [np.median(t2.d33_qs[interior]), np.median(d_eb[interior]),
                        np.median(t2.d33_cr_over_Q[interior])],
    "spread_x": [t2.d33_qs[interior].max()/t2.d33_qs[interior].min(),
                 d_eb[interior].max()/d_eb[interior].min(),
                 t2.d33_cr_over_Q[interior].max()/t2.d33_cr_over_Q[interior].min()]})
summary.round(3)
"""),
("md", """
**Lever frame.** With positions measured from x₀(t), R1 and R2 E(x) fall on one curve even though
the contact stiffened between runs (k*/k 480 → 744, notebook 05a). In the fixed 6.1 µm frame the
two runs were offset by ~18 µm.
"""),
("md", """
## Bias-resolved mode maps: piezoresponse vs electrostatic channel per position and frequency
The same two-domain fit applied at every frequency bin gives |P|(x, f) (flips with domain)
and |b|(x, f) (does not). Their ratio shows where along the lever, and at which mode, the
electrostatic force competes with the piezoresponse.
"""),
("code", """
s = runs['r2']['series']
dec = domains.decompose_series(s, band=(10e3, 1950e3))
fq = dec['freq_Hz']
# smooth over ~1 kHz so single-bin noise does not dominate the log maps
k = max(1, int(round(1e3 / np.median(np.diff(fq)))))
sm = lambda A: np.apply_along_axis(lambda r: np.convolve(r, np.ones(k)/k, 'same'), 1, A)
Pm, bm = sm(np.abs(dec['P'])), sm(np.abs(dec['b']))
fig, axs = plt.subplots(1, 3, figsize=(fp.WIDTH_IN['double'], 2.4), sharey=True)
for ax, A, lab, vr in ((axs[0], Pm, '|P| (dB re max)', (-60, 0)),
                       (axs[1], bm * 3, '|b|·3 V (dB re max |P|)', (-60, 0))):
    im = ax.pcolormesh(t2.x_lever_um, fq/1e3, (20*np.log10(np.maximum(A, 1e-30)/Pm.max())).T,
                       cmap=fp.MAP_CMAP, vmin=vr[0], vmax=vr[1], shading='nearest', rasterized=True)
    plt.colorbar(im, ax=ax, label=lab, pad=0.02)
im = axs[2].pcolormesh(t2.x_lever_um, fq/1e3, np.log10(np.maximum(bm, 1e-30)/np.maximum(Pm, 1e-30)).T,
                       cmap=fp.ERR_CMAP, vmin=-2, vmax=1, shading='nearest', rasterized=True)
plt.colorbar(im, ax=axs[2], label='log10 |b|/|P| (1/V)', pad=0.02)
for ax in axs:
    ax.set_xlabel("x − x₀, lever frame (µm)")
axs[0].set_ylabel("frequency (kHz)")
axs[0].set_title("piezoresponse"); axs[1].set_title("electrostatic"); axs[2].set_title("ratio")
fig.tight_layout(); fp.save(fig, "06_bias_resolved_maps")
"""),
("md", "## Contact potential and model checks along the lever (R1 vs R2)"),
("code", """
fig, axs = fp.figure("double", aspect=0.3, ncols=3)
for tag, t, sty in (("R1", t1, 'o:'), ("R2", t2, 'o-')):
    c = '0.55' if tag == 'R1' else fp.C['meas']
    axs[0].plot(t.x_lever_um, t.vcpd_qs, sty, color=c, ms=3, label=f'{tag} quasi-static')
    axs[0].plot(t.x_lever_um, t.vcpd_cr, sty.replace('o', 's'), color=c, ms=3, mfc='w', label=f'{tag} CR1')
    axs[1].plot(t.x_lever_um, t.ratio_qs, sty, color=c, ms=3, label=f'{tag} QS')
    axs[1].plot(t.x_lever_um, t.ratio_cr, sty.replace('o', 's'), color=c, ms=3, mfc='w', label=f'{tag} CR1')
    axs[2].plot(t.x_lever_um, t.flip_qs, sty, color=c, ms=3, label=f'{tag} QS')
    axs[2].plot(t.x_lever_um, t.flip_cr, sty.replace('o', 's'), color=c, ms=3, mfc='w', label=f'{tag} CR1')
axs[0].set_ylabel("V_cpd (V)"); axs[1].set_ylabel("|b|/|P| (1/V)"); axs[1].set_yscale('log')
axs[2].set_ylabel("domain flip angle (deg)"); axs[2].axhline(180, color='0.7', lw=0.6)
for ax in axs:
    ax.set_xlabel("x − x₀, lever frame (µm)"); ax.legend(fontsize=5.5)
fig.tight_layout(); fp.save(fig, "06_vcpd_checks")
pd.DataFrame({"run": ["R1", "R2"],
              "vcpd_qs_mean": [t1.vcpd_qs[:-1].mean(), t2.vcpd_qs[:-1].mean()],
              "ratio_cr1_mean": [t1.ratio_cr.mean(), t2.ratio_cr.mean()],
              "d33_qs_median": [np.median(t1.d33_qs), np.median(t2.d33_qs)],
              "Q_mean": [t1.Q.mean(), t2.Q.mean()]}).round(3)
"""),
("md", """
**Reading the checks.** QS flip angles sit at 181–188° and |b|/|P| at CR1 is flat along the
lever (~0.19 /V in R2), so the two-domain model holds. The CR1 flip is further from 180° than
the QS flip; the V_cpd reported in the paper comes from the QS channel and the image
extrapolation, and the free-end QS V_cpd is poorly determined because b → 0 there.
"""),
("code", """
results.save("fig6_transfer_function",
             dict(E_min=t2.E.min(), E_max=t2.E.max(), Q_mean=t2.Q.mean(), Q_sd=t2.Q.std(),
                  blind_eb_ratio_min=ratio.min(), blind_eb_ratio_max=ratio.max(),
                  d33_qs_median_interior=float(np.median(t2.d33_qs[interior])),
                  d33_cr_eb_median_interior=float(np.median(d_eb[interior])),
                  d33_cr_over_Q_median_interior=float(np.median(t2.d33_cr_over_Q[interior])),
                  d33_cr_over_Q_spread=float(summary.spread_x[2]),
                  r1_d33_qs_median=float(np.median(t1.d33_qs))),
             table=t2.assign(E_eb=eb['E_mod'], d33_cr_eb=d_eb))
"""),
]
