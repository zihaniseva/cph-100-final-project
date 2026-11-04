"""Build index.html (the illustrated report) from PROPOSAL.md.

Run from the repo root:  python figures/src/build_report.py
Needs: pip install markdown
"""
import re
from pathlib import Path

import markdown

root = Path(__file__).resolve().parents[2]
md = (root / "PROPOSAL.md").read_text(encoding="utf-8")

# body = everything from the first numbered section on
body_md = md[md.index("## 1."):]
html = markdown.markdown(body_md, extensions=["tables", "sane_lists"])

# images -> figures with a caption from the alt text
html = re.sub(r'<p><img alt="([^"]*)" src="([^"]*)" ?/?></p>',
              r'<figure><div class="fig"><img src="\2" alt="\1"></div></figure>', html)
# tables -> scrollable wrapper
html = html.replace("<table>", '<div class="tbl"><table>').replace("</table>", "</table></div>")
# numbered h2 -> styled number
html = re.sub(r"<h2>(\d+)\. ([^<]*)</h2>", r'<h2><span class="n">\1</span>\2</h2>', html)
# wrap each h2 block in a <section>
parts = re.split(r"(?=<h2>)", html)
html = "\n".join(f"<section>{p}</section>" for p in parts if p.strip())

page = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Clinical Experiencer Probing</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--bg:#f6f7f8;--surface:#fff;--fg:#1b2127;--muted:#5b6670;--rule:#dde2e6;--accent:#0f6b6e;
  --f-display:"Newsreader",Georgia,serif;--f-body:"IBM Plex Sans",system-ui,sans-serif;--f-mono:"IBM Plex Mono",ui-monospace,monospace}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#12171b;--surface:#1a2026;--fg:#e4e8eb;--muted:#97a3ad;--rule:#2c353d;--accent:#5cc2c0;color-scheme:dark}}}}
:root[data-theme="dark"]{{--bg:#12171b;--surface:#1a2026;--fg:#e4e8eb;--muted:#97a3ad;--rule:#2c353d;--accent:#5cc2c0;color-scheme:dark}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--bg);color:var(--fg);font-family:var(--f-body);font-size:16px;line-height:1.6;padding:40px 16px 72px}}
.wrap{{max-width:900px;margin:0 auto;display:flex;flex-direction:column;gap:52px}}
header{{display:flex;flex-direction:column;gap:14px;border-bottom:1px solid var(--rule);padding-bottom:28px}}
.eyebrow{{font-family:var(--f-mono);font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--accent)}}
h1{{font-family:var(--f-display);font-weight:600;font-size:clamp(2rem,5vw,2.8rem);line-height:1.1;margin:0;text-wrap:balance}}
.lede{{font-size:1.12rem;max-width:66ch;margin:0}}
.meta{{display:flex;flex-wrap:wrap;gap:6px 22px;font-family:var(--f-mono);font-size:12.5px;color:var(--muted)}}
.meta b{{color:var(--fg);font-weight:500}}
section{{display:flex;flex-direction:column;gap:16px}}
h2{{font-family:var(--f-display);font-weight:600;font-size:1.7rem;line-height:1.2;margin:0;display:flex;gap:12px;align-items:baseline}}
h2 .n{{font-family:var(--f-mono);font-size:.8rem;color:var(--muted);font-weight:500}}
h3{{font-size:1.02rem;font-weight:600;margin:10px 0 0}}
p,ul,ol{{margin:0;max-width:70ch}}
ul,ol{{padding-left:1.2em;display:flex;flex-direction:column;gap:6px}}
a{{color:var(--accent)}}
code{{font-family:var(--f-mono);font-size:.88em}}
figure{{margin:6px 0}}
.fig{{background:#fff;border:1px solid var(--rule);border-radius:8px;padding:8px;overflow:hidden}}
.fig img{{display:block;width:100%;height:auto}}
.tbl{{overflow-x:auto;border:1px solid var(--rule);border-radius:8px;background:var(--surface)}}
table{{border-collapse:collapse;width:100%;font-size:14.5px}}
th,td{{padding:9px 12px;text-align:left;border-bottom:1px solid var(--rule);vertical-align:top}}
th{{font-weight:600;font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.05em}}
tr:last-child td{{border-bottom:0}}
</style>
</head>
<body><div class="wrap">
<header>
  <div class="eyebrow">CPHC100 Final Project · Progress report</div>
  <h1>Decoding clinical experiencer information in a small language model</h1>
  <p class="lede">At which layers can we decode whether a disease belongs to the patient or a family member, and does that information generalize across wording?</p>
  <div class="meta"><span><b>Author</b> Eva (zihaniseva)</span><span><b>Partner</b> none, working solo</span><span><b>Date</b> November 3, 2026</span><span><a href="https://github.com/zihaniseva/cph-100-final-project">Code and data on GitHub</a></span></div>
</header>
{html}
</div></body></html>
"""
(root / "index.html").write_text(page, encoding="utf-8")
print("wrote", root / "index.html")
