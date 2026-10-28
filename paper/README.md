# Manuscript (AI-written)

AI-written report in the style of a PLOS Computational Biology research article, with supporting information. It has not been checked by a human; see the disclaimer in the repository README.

- `manuscript.md` — source (citations as `{key}`; resolved by `build.py` from `references_verified.md`)
- `manuscript.docx` — built manuscript (double spacing, continuous line numbers, numbered references, captions after first citing paragraph)
- `S1_Text` — original task prompt; `S2_Text` — methods that did not work; `S3_Text` — agent log
- `figures/` — Fig1–Fig5 (TIFF for submission, PNG previews) and captions

Rebuild (requires pandoc and python-docx): `python build.py` from this folder (expects `figures/captions.md`).
