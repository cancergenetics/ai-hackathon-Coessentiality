"""Build docs/explainer/explainer.html (one self-contained interactive page) from the Markdown explainer.

Sources: METHOD.md, JOURNEY.md, INSIGHTS.md, figures/*, data/journey.json. Markdown is converted with pandoc
(set PANDOC or have pandoc on PATH). Markers in the Markdown are replaced by interactive widgets:
  <!-- widget:pipeline -->   + the image paragraph after it  -> clickable pipeline diagram
  <!-- widget:timeline -->   + the image paragraph after it  -> journey timeline chart
  <!-- worked:start --> ... <!-- worked:end -->              -> step-by-step worked example (one h4 per step)
Turning points (h2 headings preceded by <a id="tpN"></a>) become expandable <details> blocks.
Usage: PANDOC=~/anaconda3/bin/pandoc ./.env/bin/python docs/explainer/build_html.py
"""
import base64
import html
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
PANDOC = os.environ.get("PANDOC") or shutil.which("pandoc") or str(Path.home() / "anaconda3/bin/pandoc")
TABS = [("method", "Method", "METHOD.md"), ("journey", "How we got here", "JOURNEY.md"), ("insights", "Insights", "INSIGHTS.md")]


def md_to_html(path: Path) -> str:
    out = subprocess.run([PANDOC, "--from", "gfm", "--to", "html5", "--wrap=none", str(path)],
                         capture_output=True, text=True, check=True).stdout
    return out


def inline_images(h: str) -> str:
    def rep(m):
        src = m.group(1)
        p = HERE / src
        if not p.exists() or src.startswith(("http", "data:")):
            return m.group(0)
        mime = {"png": "image/png", "svg": "image/svg+xml"}[p.suffix[1:]]
        return f'src="data:{mime};base64,{base64.b64encode(p.read_bytes()).decode()}"'
    return re.sub(r'src="([^"]+)"', rep, h)


def fix_links(h: str) -> str:
    """Links between the Markdown files become in-page anchors; links into the repo go to GitHub."""
    h = re.sub(r'href="(?:METHOD|JOURNEY|INSIGHTS|README)\.md#([^"]+)"', r'href="#\1"', h)
    h = re.sub(r'href="(?:METHOD|JOURNEY|INSIGHTS|README)\.md"', 'href="#top"', h)

    def repo(m):
        target = m.group(1)
        if target.startswith(("http", "#", "data:", "mailto")):
            return m.group(0)
        path = os.path.normpath(os.path.join("docs/explainer", target.split("#")[0]))
        return f'href="https://github.com/cancergenetics/ai-hackathon-Coessentiality/blob/main/{path}"'
    return re.sub(r'href="([^"]+)"', repo, h)


def widgets(h: str) -> str:
    svg = (HERE / "figures/pipeline.svg").read_text()
    svg = re.sub(r"<\?xml[^>]*>", "", svg)
    h = re.sub(r"<!-- widget:pipeline -->\s*<p>.*?</p>", f'<figure class="pipeline" id="pipeline">{svg}'
               '<figcaption>Click a stage to jump to its explanation.</figcaption></figure>', h, flags=re.S)
    h = re.sub(r"<!-- widget:timeline -->\s*<p>.*?</p>(\s*<p><em>.*?</em></p>)?", '<figure class="timeline" id="timeline"></figure>', h, flags=re.S)

    def worked(m):
        body = m.group(1)
        parts = re.split(r"(?=<h4)", body)
        intro, steps = parts[0], [p for p in parts[1:] if p.strip()]
        panes = []
        for i, p in enumerate(steps):
            title = re.search(r"<h4[^>]*>(.*?)</h4>", p, re.S).group(1)
            content = re.sub(r"<h4[^>]*>.*?</h4>", "", p, count=1, flags=re.S)
            panes.append((title, content))
        nav = "".join(f'<button type="button" class="wk-tab" id="wk-tab-{i}" data-i="{i}" aria-controls="wk-pane-{i}">'
                      f'<span class="wk-n">{i}</span><span class="wk-t">{re.sub("<[^>]+>", "", t).split(". ", 1)[-1]}</span></button>'
                      for i, (t, _) in enumerate(panes))
        body = "".join(f'<section class="wk-pane" id="wk-pane-{i}" data-i="{i}"{"" if i == 0 else " hidden"}>'
                       f'<h4>{t}</h4>{c}</section>' for i, (t, c) in enumerate(panes))
        return (f'{intro}<div class="worked" id="worked"><div class="wk-nav" role="tablist">{nav}</div>'
                f'<div class="wk-body">{body}</div><div class="wk-ctrl"><button type="button" id="wk-prev">← Previous</button>'
                f'<span id="wk-pos"></span><button type="button" id="wk-next">Next →</button></div></div>')
    return re.sub(r"<!-- worked:start -->(.*?)<!-- worked:end -->", worked, h, flags=re.S)


def turning_points(h: str) -> str:
    """<a id="tpN"></a><h2>..</h2> ... up to the next anchor or h2 -> <details class="tp" id="tpN">."""
    pat = re.compile(r'<p><a id="(tp\d+)"></a></p>\s*<h2[^>]*>(.*?)</h2>(.*?)(?=<p><a id="|<h2|\Z)', re.S)
    return pat.sub(lambda m: f'<details class="tp" id="{m.group(1)}"><summary><span class="tp-title">{m.group(2)}</span>'
                   f'</summary><div class="tp-body">{m.group(3)}</div></details>', h)


def main():
    sections = []
    for key, label, fname in TABS:
        h = md_to_html(HERE / fname)
        h = re.sub(r"<h1[^>]*>.*?</h1>", "", h, count=1, flags=re.S)
        h = widgets(h)
        if key == "journey":
            h = turning_points(h)
        h = fix_links(inline_images(h))
        sections.append((key, label, h))
    journey = json.loads((HERE / "data/journey.json").read_text())
    worked = json.loads((HERE / "data/worked_example.json").read_text())
    tpl = (HERE / "template.html").read_text()
    tabs = "".join(f'<button type="button" role="tab" class="tab" id="tab-{k}" data-tab="{k}" aria-controls="panel-{k}">{l}</button>'
                   for k, l, _ in sections)
    panels = "".join(f'<section role="tabpanel" class="panel" id="panel-{k}" data-tab="{k}"{"" if i == 0 else " hidden"}>'
                     f'<div class="prose">{h}</div></section>' for i, (k, _, h) in enumerate(sections))
    page = (tpl.replace("{{TABS}}", tabs).replace("{{PANELS}}", panels)
            .replace("{{JOURNEY_JSON}}", json.dumps(journey))
            .replace("{{WORKED_JSON}}", json.dumps({"rt": worked.get("runtime_s")})))
    (HERE / "explainer.html").write_text(page)
    print(f"wrote {HERE / 'explainer.html'} ({len(page) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
