import numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt

d = np.load('stage5_raw.npz')
Msk = np.load('stage5_mask.npz'); M1, M2 = Msk['M1'], Msk['M2']

INK, SEC, MUTED, GRID, SURF = '#0b0b0b', '#52514e', '#898781', '#e1e0d9', '#fcfcfb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 9, 'axes.edgecolor': '#c3c2b7',
                     'axes.labelcolor': SEC, 'xtick.color': MUTED, 'ytick.color': MUTED,
                     'axes.titlecolor': INK, 'figure.facecolor': SURF, 'axes.facecolor': SURF,
                     'legend.frameon': False})
ext = [0, 6, 0, 6]

# per-domain, in-situ enhancement (CR1(0V) / qs(0V) at that position, this session, this tip)
E_MAP = {
    'A': {'E1': 198.2, 'E2': 225.8},   # position A: away from the node, matches stage-1's 200.2
    'B': {'E1': 25.3,  'E2': 18.1},    # position B: node shifted after the load ladder -- NOT 2.8
}
VAC = 1.0

def d33_map(name, pos):
    amp = d[f'{name}_amp'] * 1e12   # pm
    is_qs = 'qs' in name
    if is_qs:
        return amp / VAC             # already quantitative -- no E involved
    E1, E2 = E_MAP[pos]['E1'], E_MAP[pos]['E2']
    out = np.full_like(amp, np.nan)
    out[M1] = amp[M1] / (VAC * E1)
    out[M2] = amp[M2] / (VAC * E2)
    return out

rows = [
    ('A', 'position A (148.8 µm)', 'A_qs20k_0V', 'A_cr1_0V', 'A_cr1_Vcpd'),
    ('B', 'position B (free end)', 'B_qs20k_0V', 'B_cr1_0V', 'B_cr1_Vcpd'),
]
VMAX = 15  # pm/V, shared scale so panels are genuinely comparable

fig, axs = plt.subplots(2, 3, figsize=(12.2, 7.8))
for r, (pos, lab, qsname, cr0, crv) in enumerate(rows):
    m_qs = d33_map(qsname, pos)
    m_cr0 = d33_map(cr0, pos)
    m_crv = d33_map(crv, pos)
    for c, (title, m) in enumerate([
        ('quasi-static d\N{LATIN SUBSCRIPT SMALL LETTER I}\N{LATIN SUBSCRIPT SMALL LETTER I} (pm/V)\n[0 V, no E needed]', m_qs),
        ('CR1 d33 (pm/V)\n[0 V, ÷ in-situ E per domain]', m_cr0),
        ('CR1 d33 (pm/V)\n[+0.77 V, ÷ in-situ E per domain]', m_crv),
    ]):
        ax = axs[r, c]
        im = ax.imshow(m, extent=ext, cmap='magma', vmin=0, vmax=VMAX)
        ax.set_title((f'{lab}\n' if c == 0 else '') + title, loc='left', fontsize=8.5)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03); cb.set_label('pm/V', fontsize=7); cb.ax.tick_params(labelsize=7)

for ax in axs[-1, :]:
    ax.set_xlabel('µm')
for ax in axs[:, 0]:
    ax.set_ylabel('µm')

fig.suptitle('Genuinely quantitative d33 maps (pm/V, common 0-15 pm/V scale) -- CR1 amplitude divided by\n'
             'V_ac × the in-situ enhancement measured at each domain, not the raw |Z| shown previously',
             fontsize=11, color=INK, x=0.01, ha='left')
fig.tight_layout(rect=(0, 0, 1, 0.90))
fig.savefig('figK_quantitative_d33_maps.png', dpi=150)
print('saved figK')

# quick sanity print: median d33 per domain per frame
for pos, lab, qsname, cr0, crv in rows:
    for name in (qsname, cr0, crv):
        m = d33_map(name, pos)
        print(f'{pos} {name:16s} domain1={np.nanmedian(m[M1]):.2f}  domain2={np.nanmedian(m[M2]):.2f} pm/V')
