"""Build manuscript.docx and S1_Text.docx from Markdown sources.

- {key} / {key1, key2} citations -> [n] / [n,m] numbered by first appearance (Vancouver, PLOS style)
- {REFERENCES} -> numbered list from references_verified.md (only cited items)
- {FIGCAPTIONS} -> figures/captions.md
- pandoc -> DOCX; then double spacing, continuous line numbers, page numbers via python-docx.
Usage: ../.env/bin/python build.py   (run from manuscript/)
"""
import os
import re
import shutil
import subprocess
from pathlib import Path

import docx
from docx.enum.text import WD_LINE_SPACING
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

HERE = Path(__file__).resolve().parent
PANDOC = os.environ.get("PANDOC") or shutil.which("pandoc") or "pandoc"
KEY = r"[a-z]+[0-9]{4}[a-z0-9]*"
CITE = re.compile(r"(?<![_^])\{(" + KEY + r"(?:\s*,\s*" + KEY + r")*)\}")


def load_refs():
    refs = {}
    for line in (HERE / "references_verified.md").read_text().splitlines():
        cells = [c.strip() for c in line.strip().strip("|").split("|")]
        if len(cells) >= 3 and re.fullmatch(r"[a-z][a-z0-9]+", cells[0]) and cells[0] != "key":
            doi = cells[2].strip()
            has_doi = re.search(r"\bdoi:", cells[1], re.I) is not None
            ok = doi and doi.lower() not in ("n/a", "-", "none") and not doi.lower().startswith(("none", "unverified", "no doi"))
            refs[cells[0]] = cells[1] + (f" doi:{doi}" if ok and not has_doi else "")
    return refs


def ranges(nums):
    nums = sorted(set(nums)); out = []; i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        out.append(f"{nums[i]}–{nums[j]}" if j - i >= 2 else ",".join(str(n) for n in nums[i:j + 1]))
        i = j + 1
    return ",".join(out)


def number_citations(text, refs, order):
    def sub(m):
        keys = [k.strip() for k in m.group(1).split(",")]
        if not all(k in refs for k in keys):
            missing = [k for k in keys if k not in refs]
            if keys[0].isupper() or m.group(1) in ("REFERENCES", "FIGCAPTIONS"):
                return m.group(0)
            raise KeyError(f"unknown citation keys {missing}")
        for k in keys:
            if k not in order:
                order.append(k)
        return "[" + ranges([order.index(k) + 1 for k in keys]) + "]"
    return CITE.sub(sub, text)


def inline_captions(text, captions):
    """PLOS: each figure caption goes directly after the paragraph that first cites the figure."""
    blocks = re.split(r"(?m)^(?=\*\*Fig \d+\. )", captions.strip())
    blocks = {int(re.match(r"\*\*Fig (\d+)\.", b).group(1)): b.strip() for b in blocks if b.strip().startswith("**Fig")}
    paras = text.split("\n\n")
    body_end = next(i for i, p in enumerate(paras) if p.startswith("# Data availability"))
    placed = {}
    for i, p in enumerate(paras[:body_end]):
        if p.startswith("#") or p.startswith("**Fig"):
            continue
        for n in re.findall(r"\bFigs? (\d+)", p):
            n = int(n)
            if n in blocks and n not in placed:
                placed[n] = i
    missing = sorted(set(blocks) - set(placed))
    if missing:
        raise ValueError(f"figures never cited in body: {missing}")
    for n, i in sorted(placed.items(), key=lambda kv: (-kv[1], -kv[0])):
        paras.insert(i + 1, blocks[n])
    return "\n\n".join(paras)


def format_docx(path, line_numbers=True):
    d = docx.Document(path)
    for p in d.paragraphs:
        p.paragraph_format.line_spacing_rule = WD_LINE_SPACING.DOUBLE
    for sec in d.sections:
        sp = sec._sectPr
        if line_numbers:
            ln = OxmlElement("w:lnNumType")
            ln.set(qn("w:countBy"), "1"); ln.set(qn("w:restart"), "continuous")
            sp.append(ln)
        footer = sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
        run = footer.add_run()
        for tag, txt in (("begin", None), (None, "PAGE"), ("end", None)):
            if tag:
                el = OxmlElement("w:fldChar"); el.set(qn("w:fldCharType"), tag)
            else:
                el = OxmlElement("w:instrText"); el.set(qn("xml:space"), "preserve"); el.text = txt
            run._r.append(el)
    d.save(path)


def build(src, out, refs, captions=None):
    order = []
    text = (HERE / src).read_text()
    text = number_citations(text, refs, order)
    if "{REFERENCES}" in text:
        text = text.replace("{REFERENCES}", "\n".join(f"{i}. {refs[k]}" for i, k in enumerate(order, 1)))
    if captions is not None:
        text = inline_captions(text, captions)
    md = HERE / (Path(out).stem + ".built.md")
    md.write_text(text)
    subprocess.run([PANDOC, str(md), "-o", str(HERE / out)], check=True)
    format_docx(HERE / out)
    return order


if __name__ == "__main__":
    refs = load_refs()
    cap_file = HERE / "figures" / "captions.md"
    caps = cap_file.read_text() if cap_file.exists() else "[figure captions pending]"
    order = build("manuscript.md", "manuscript.docx", refs, caps)
    print(f"manuscript.docx: {len(order)} references cited")
    if (HERE / "S1_Text.md").exists():
        build("S1_Text.md", "S1_Text.docx", refs)
        print("S1_Text.docx built")
    for name in ("S2_Text", "S3_Text"):
        if (HERE / f"{name}.md").exists():
            build(f"{name}.md", f"{name}.docx", refs)
            print(f"{name}.docx built")
