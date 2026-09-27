# Deck build

`build_deck.py` assembles the 21-slide ORNL deck from `../results/fig/*.png`.
It expects the ORNL template unpacked beside it as `ornl.pptx` (copy
`ORNL template 1.pptm` to `ornl.pptx`; the script's `demacro()` rewrites the
macroEnabled content type after save — without that fix PowerPoint refuses the
output; see the note inside).

QA chain used: `validate.py out.pptx --original ornl.pptx`, a check that
`[Content_Types].xml` no longer contains `macroEnabled`, LibreOffice render to
JPGs, and a visual pass over every slide.
