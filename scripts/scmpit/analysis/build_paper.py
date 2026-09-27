"""Build the manuscript Word document from paper.md + paper_tail.md."""
import os
import re
import subprocess

SRC = '/home/claude/scmpit_analysis'
HERE = '/home/claude/docbuild'
OUT = '/mnt/user-data/outputs'

FIGS = [
    ('Figure 1', 'paper_fig1_transfer.png',
     'Figure 1. Recovery of the cantilever transfer function from the Run 2 dataset. '
     '(a) measured mode shapes and nodes; (b) frequency flatness over the 3.2 h map; '
     '(c) the measured enhancement against Q; (d) the static shape and its clamped-beam fit; '
     '(e) the two channels along the beam; (f) low-rank wideband reconstruction.'),
    ('Figure 2', 'paper_fig2_load.png',
     'Figure 2. The load dimension, 79 minutes. (a) CR1 frequency and Q versus load; '
     '(b) the measured frequencies placed on the model f(k*) curve, showing saturation; '
     '(c) the enhancement along the lever at each load; (d) contact potential and d33 versus load.'),
    ('Figure 3', 'paper_fig3_cpd.png',
     'Figure 3. Contact potential. (a) bias dependence at the working position; '
     '(b) V_cpd along the beam in both channels; (c) the four independent determinations; '
     '(d) domain amplitude ratio versus bias, both campaigns.'),
    ('Figure 4', 'paper_fig4_maps.png',
     'Figure 4. Quantitative PFM maps. Every panel is amplitude divided by '
     '(V_ac x the in-situ enhancement for that domain), on one common scale.'),
    ('Figure 5', 'paper_fig5_linearity.png',
     'Figure 5. Drive-amplitude linearity, the additive detection floor, and the case for '
     'contact resonance.'),
    ('Figure 6', 'paper_fig6_eb_d33.png',
     'Figure 6. The blind Euler-Bernoulli fit. (a) the enhancement predicted with every '
     'amplitude withheld; (b) d33 versus position for the quasi-static and CR1 channels; '
     '(c) each route against the model-free answer.'),
    ('Figure S1', 'paper_figS1_null_comparison.png',
     'Figure S1. Does the bias null equalise the domains? Position A, identical frame and scale, '
     'both campaigns.'),
]


def inline_figures(md):
    """Insert each figure once, after the paragraph that first cites it."""
    paras = md.split('\n\n')
    placed, out = set(), []
    for p in paras:
        out.append(p)
        if p.lstrip().startswith('|') or p.lstrip().startswith('**Table'):
            continue
        for tag, fn, cap in FIGS:
            if tag in placed or not re.search(re.escape(tag) + r'(?![0-9])', p):
                continue
            if not os.path.exists(os.path.join(SRC, fn)):
                print(f'  MISSING {fn}')
                continue
            placed.add(tag)
            out.append(f'![{cap}]({SRC}/{fn}){{width=6.4in}}')
    missing = [t for t, _, _ in FIGS if t not in placed]
    if missing:
        print('  never cited, appended at the end:', missing)
    return '\n\n'.join(out), placed


def manual_toc(md):
    items = []
    for ln in md.split('\n'):
        m = re.match(r'^(#{2,3}) (.+)$', ln)
        if m:
            depth = len(m.group(1))
            text = re.sub(r'[*`$\\]', '', m.group(2)).strip()
            items.append(' ' * (6 * (depth - 2)) + text)
    return '## Contents\n\n' + '\\\n'.join(items) + '\n\n---\n\n' if items else ''


head = open(os.path.join(SRC, 'paper.md'), encoding='utf-8').read()
tail = open(os.path.join(SRC, 'paper_tail.md'), encoding='utf-8').read()
# paper.md ends in stub headings for the sections paper_tail.md actually carries
head = head[:head.index('\n## 4. Discussion')].rstrip().rstrip('-').rstrip()
md = head + '\n\n---\n\n' + tail
md = re.sub(r'\A# [^\n]*\n', '', md)
# the byline is carried by the YAML author field
md = re.sub(r'\A\s*\*\*Liam Collins\*\*[^\n]*\n', '', md.lstrip())
md = md.lstrip('-\n ')

title = ('Recovering the cantilever transfer function in situ: quantitative on-resonance '
         'piezoresponse force microscopy with same-session contact-potential and d33 calibration')
meta = (f'---\ntitle: "{title}"\n'
        f'author: "Liam Collins — Center for Nanophase Materials Sciences, '
        f'Oak Ridge National Laboratory"\ndate: "2026-09-21"\n---\n\n')

toc = manual_toc(md)
md, used = inline_figures(md)
first = md.find('\n## 1. Introduction')
md = md[:first + 1] + toc + md[first + 1:] if first > 0 else toc + md

tmp = os.path.join(HERE, 'paper_doc.md')
open(tmp, 'w', encoding='utf-8').write(meta + md)
dst = os.path.join(OUT, 'SCMPIT_Quantitative_CR_PFM_manuscript_2026-09-21.docx')
subprocess.run(['pandoc', tmp, '-o', dst, '--reference-doc', os.path.join(HERE, 'ref.docx'),
                '--from', 'markdown+pipe_tables+tex_math_dollars', '--resource-path', SRC],
               check=True)
print(f'  {os.path.basename(dst)}: {len(used)} figures, {os.path.getsize(dst)/1e6:.2f} MB')
