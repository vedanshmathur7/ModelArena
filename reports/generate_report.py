"""Convert report.md to a styled HTML file ready to print-to-PDF."""
from __future__ import annotations

import markdown
import pathlib

REPORTS_DIR = pathlib.Path(__file__).resolve().parent

md_text = (REPORTS_DIR / "report.md").read_text(encoding="utf-8")

css = """
@page {
    size: A4;
    margin: 8mm;
}
body {
    font-family: Arial, Helvetica, sans-serif;
    max-width: 980px;
    margin: 0 auto;
    padding: 0 10px;
    line-height: 1.22;
    color: #172033;
    font-size: 10px;
}
h1 {
    color: #172033;
    border-bottom: 3px solid #4C72B0;
    padding-bottom: 5px;
    margin: 4px 0 5px;
    font-size: 19px;
}
h2 {
    color: #172033;
    border-bottom: 1px solid #ddd;
    padding-bottom: 3px;
    margin: 8px 0 5px;
    font-size: 13px;
}
p { margin: 3px 0; }
ul { margin: 3px 0 5px 14px; padding: 0; }
li { margin: 2px 0; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 5px 0;
    font-size: 9px;
}
td, th {
    border: 1px solid #ccc;
    padding: 3px 5px;
    text-align: left;
}
th { background: #4C72B0; color: #fff; }
tr:nth-child(even) { background: #f9f9f9; }
code {
    background: #f0f0f0;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 11px;
}
img {
    display: block;
    max-width: 68%;
    max-height: 130px;
    object-fit: contain;
    margin: 4px auto 5px;
}
strong {
    color: #172033;
}
@media print {
    body { font-size: 10px; }
    h1 { font-size: 19px; }
    h2 { font-size: 13px; }
    img { max-height: 125px; }
}
"""

body = markdown.markdown(md_text, extensions=["tables", "fenced_code"])

html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>Evaluation Report — OSS vs Frontier AI Assistant</title>
  <style>{css}</style>
</head>
<body>
{body}
</body>
</html>"""

out = REPORTS_DIR / "evaluation_report.html"
out.write_text(html, encoding="utf-8")
print(f"Generated: {out.resolve()}")
print("Open in your browser → Ctrl+P (or Cmd+P) → Save as PDF")
