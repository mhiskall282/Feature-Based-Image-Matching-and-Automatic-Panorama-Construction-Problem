"""
scripts/generate_submission_pdfs.py
===================================
Generates publication-grade PDF documents for the CSCD608 Examination Submission:
1. report/CSCD608_Examination_Project_Report.pdf (Formal Academic Research Paper)
2. docs/CSCD608_Technical_Documentation_Manual.pdf (Complete System Documentation)
3. report/CSCD608_Submission_Portfolio.pdf (Combined Master Examination Package)
"""

import os
import sys
import subprocess
import time
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).resolve().parent.parent
REPORT_MD = ROOT_DIR / "report" / "report.md"
DOCS_MD = ROOT_DIR / "docs" / "README.md"

REPORT_PDF = ROOT_DIR / "report" / "CSCD608_Examination_Project_Report.pdf"
REPORT_PDF_ALIAS = ROOT_DIR / "report" / "report.pdf"
DOCS_PDF = ROOT_DIR / "docs" / "CSCD608_Technical_Documentation_Manual.pdf"
DOCS_PDF_ALIAS = ROOT_DIR / "docs" / "documentation.pdf"

# Find browser executable
def find_chrome():
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None


def markdown_to_html(md_text: str, title: str, subtitle: str, doc_type: str = "report") -> str:
    """Convert Markdown content to a beautifully styled, print-ready HTML page."""
    import re
    
    html_body = []
    in_code_block = False
    code_lang = ""
    code_lines = []
    
    in_table = False
    table_rows = []
    
    for line in md_text.splitlines():
        # Code block handling
        if line.strip().startswith("```"):
            if in_code_block:
                in_code_block = False
                code_content = "\n".join(code_lines)
                if code_lang == "mermaid":
                    html_body.append(f'<div class="mermaid-diagram"><pre class="mermaid">{code_content}</pre></div>')
                else:
                    html_body.append(f'<pre class="code-block"><code>{code_content}</code></pre>')
                code_lines = []
            else:
                in_code_block = True
                code_lang = line.strip()[3:].strip()
            continue
            
        if in_code_block:
            code_lines.append(line)
            continue
            
        # Table handling
        if "|" in line and not line.strip().startswith("#"):
            if not in_table:
                in_table = True
                table_rows = [line]
            else:
                table_rows.append(line)
            continue
        elif in_table:
            in_table = False
            # Render table
            if len(table_rows) >= 2:
                th_cells = [c.strip() for c in table_rows[0].split("|") if c.strip()]
                table_html = ['<div class="table-container"><table class="academic-table"><thead><tr>']
                for th in th_cells:
                    table_html.append(f'<th>{th}</th>')
                table_html.append('</tr></thead><tbody>')
                
                for row in table_rows[2:]:
                    td_cells = [c.strip() for c in row.split("|") if c.strip()]
                    if td_cells:
                        table_html.append('<tr>')
                        for td in td_cells:
                            # format bold / code in td
                            td_f = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', td)
                            td_f = re.sub(r'`(.*?)`', r'<code>\1</code>', td_f)
                            table_html.append(f'<td>{td_f}</td>')
                        table_html.append('</tr>')
                table_html.append('</tbody></table></div>')
                html_body.append("".join(table_html))
            table_rows = []

        # Headers
        if line.startswith("# "):
            continue # Title handled in header
        elif line.startswith("## "):
            h2_text = line[3:].strip()
            html_body.append(f'<h2 class="section-title">{h2_text}</h2>')
        elif line.startswith("### "):
            h3_text = line[4:].strip()
            html_body.append(f'<h3 class="subsection-title">{h3_text}</h3>')
        elif line.startswith("#### "):
            h4_text = line[5:].strip()
            html_body.append(f'<h4 class="subsubsection-title">{h4_text}</h4>')
        elif line.startswith("> [!"):
            callout_type = line[4:].split("]")[0].strip().lower()
            html_body.append(f'<div class="callout callout-{callout_type}">')
        elif line.startswith("> "):
            html_body.append(f'<p class="callout-text">{line[2:].strip()}</p>')
        elif line.strip().startswith("- ") or line.strip().startswith("* "):
            item_text = line.strip()[2:].strip()
            item_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item_text)
            item_text = re.sub(r'`(.*?)`', r'<code>\1</code>', item_text)
            html_body.append(f'<li class="list-item">{item_text}</li>')
        elif re.match(r'^\d+\.\s', line.strip()):
            item_text = re.sub(r'^\d+\.\s', '', line.strip())
            item_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', item_text)
            item_text = re.sub(r'`(.*?)`', r'<code>\1</code>', item_text)
            html_body.append(f'<li class="list-item-ordered">{item_text}</li>')
        elif line.strip() == "---":
            html_body.append('<hr class="divider">')
        elif line.strip():
            p_text = line.strip()
            p_text = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', p_text)
            p_text = re.sub(r'`(.*?)`', r'<code>\1</code>', p_text)
            html_body.append(f'<p class="paragraph">{p_text}</p>')

    body_content = "\n".join(html_body)

    template = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>{title}</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

  @page {{
    size: A4;
    margin: 20mm 18mm 20mm 18mm;
    @bottom-right {{
      content: counter(page);
    }}
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    color: #1e293b;
    background-color: #ffffff;
    line-height: 1.6;
    font-size: 10pt;
  }}

  .header-cover {{
    border-bottom: 3px solid #2563eb;
    padding-bottom: 1.25rem;
    margin-bottom: 2rem;
  }}

  .institution {{
    font-size: 11pt;
    font-weight: 800;
    color: #2563eb;
    text-transform: uppercase;
    letter-spacing: 0.08em;
  }}

  .doc-title {{
    font-size: 18pt;
    font-weight: 800;
    color: #0f172a;
    margin-top: 0.4rem;
    margin-bottom: 0.4rem;
    line-height: 1.25;
  }}

  .doc-subtitle {{
    font-size: 11pt;
    font-weight: 600;
    color: #475569;
    margin-bottom: 0.8rem;
  }}

  .meta-grid {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.5rem;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    padding: 0.6rem 0.8rem;
    font-size: 8.5pt;
  }}

  .meta-item strong {{
    color: #0f172a;
  }}

  .section-title {{
    font-size: 13pt;
    font-weight: 800;
    color: #0f172a;
    border-bottom: 1.5px solid #cbd5e1;
    padding-bottom: 0.3rem;
    margin-top: 1.5rem;
    margin-bottom: 0.75rem;
    page-break-after: avoid;
  }}

  .subsection-title {{
    font-size: 11pt;
    font-weight: 700;
    color: #1e40af;
    margin-top: 1.1rem;
    margin-bottom: 0.45rem;
    page-break-after: avoid;
  }}

  .subsubsection-title {{
    font-size: 10pt;
    font-weight: 700;
    color: #334155;
    margin-top: 0.8rem;
    margin-bottom: 0.35rem;
    page-break-after: avoid;
  }}

  .paragraph {{
    margin-bottom: 0.75rem;
    text-align: justify;
  }}

  .list-item, .list-item-ordered {{
    margin-left: 1.4rem;
    margin-bottom: 0.35rem;
  }}

  .table-container {{
    margin: 1rem 0;
    page-break-inside: avoid;
  }}

  .academic-table {{
    width: 100%;
    border-collapse: collapse;
    font-size: 8.5pt;
  }}

  .academic-table th {{
    background-color: #f1f5f9;
    color: #0f172a;
    font-weight: 700;
    border: 1px solid #cbd5e1;
    padding: 6px 8px;
    text-align: left;
  }}

  .academic-table td {{
    border: 1px solid #e2e8f0;
    padding: 5px 8px;
    color: #334155;
  }}

  .academic-table tr:nth-child(even) {{
    background-color: #f8fafc;
  }}

  .code-block {{
    background-color: #f8fafc;
    border: 1px solid #e2e8f0;
    border-left: 3px solid #3b82f6;
    border-radius: 4px;
    padding: 8px 12px;
    font-family: 'JetBrains Mono', Consolas, monospace;
    font-size: 8pt;
    margin: 0.8rem 0;
    white-space: pre-wrap;
    page-break-inside: avoid;
    color: #0f172a;
  }}

  code {{
    font-family: 'JetBrains Mono', Consolas, monospace;
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    font-size: 8.5pt;
    color: #0f172a;
  }}

  .divider {{
    border: 0;
    height: 1px;
    background: #e2e8f0;
    margin: 1.5rem 0;
  }}

  .callout {{
    border-left: 4px solid #3b82f6;
    background: #eff6ff;
    padding: 0.6rem 0.9rem;
    border-radius: 0 6px 6px 0;
    margin: 0.9rem 0;
    page-break-inside: avoid;
  }}

  .callout-text {{
    font-size: 9pt;
    color: #1e3a8a;
  }}

  .footer-note {{
    margin-top: 2rem;
    border-top: 1px solid #e2e8f0;
    padding-top: 0.5rem;
    font-size: 8pt;
    color: #64748b;
    display: flex;
    justify-content: space-between;
  }}
</style>
<!-- MathJax for rendering LaTeX math formulas cleanly -->
<script>
  window.MathJax = {{
    tex: {{
      inlineMath: [['$', '$']],
      displayMath: [['$$', '$$']]
    }},
    svg: {{
      fontCache: 'global'
    }}
  }};
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>

<div class="header-cover">
  <div class="institution">University of Ghana • Department of Computer Science</div>
  <h1 class="doc-title">{title}</h1>
  <div class="doc-subtitle">{subtitle}</div>
  <div class="meta-grid">
    <div class="meta-item"><strong>Course:</strong> CSCD608: Advanced Computer Vision (3 Credits)</div>
    <div class="meta-item"><strong>Examination:</strong> 2025/2026 Second Semester</div>
    <div class="meta-item"><strong>Question:</strong> Question 1 — Feature Matching & Panoramic Stitching</div>
    <div class="meta-item"><strong>Author:</strong> Postgraduate Candidate (MPhil / MSc Computer Science)</div>
  </div>
</div>

{body_content}

<div class="footer-note">
  <span>CSCD608: Advanced Computer Vision — Examination Project</span>
  <span>Department of Computer Science</span>
</div>

</body>
</html>"""
    return template


def convert_html_to_pdf(html_path: Path, pdf_path: Path):
    """Render HTML to PDF using headless Chrome/Edge."""
    chrome_exe = find_chrome()
    if not chrome_exe:
        print("ERROR: No Chrome or Edge executable found.")
        return False
        
    cmd = [
        chrome_exe,
        "--headless",
        "--disable-gpu",
        "--no-pdf-header-footer",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        str(html_path.resolve()),
    ]
    
    print(f"Generating PDF: {pdf_path.name}...")
    res = subprocess.run(cmd, capture_output=True)
    if res.returncode == 0 and pdf_path.exists():
        print(f"SUCCESS: Created {pdf_path.name} ({pdf_path.stat().st_size / 1024:.1f} KB)")
        return True
    else:
        print(f"Failed to generate {pdf_path.name}: {res.stderr.decode()}")
        return False


def main():
    print("\n" + "="*70)
    print("CSCD608 SUBMISSION PDF GENERATOR")
    print("="*70 + "\n")
    
    # 1. Generate Examination Report PDF
    if REPORT_MD.exists():
        report_text = REPORT_MD.read_text(encoding="utf-8")
        report_html = markdown_to_html(
            report_text,
            title="Feature-Based Image Matching & Automatic Panorama Construction",
            subtitle="CSCD608: Advanced Computer Vision — Formal Project Examination Report",
            doc_type="report"
        )
        report_html_path = ROOT_DIR / "report" / "report_print.html"
        report_html_path.write_text(report_html, encoding="utf-8")
        
        convert_html_to_pdf(report_html_path, REPORT_PDF)
        if REPORT_PDF.exists():
            import shutil
            shutil.copyfile(REPORT_PDF, REPORT_PDF_ALIAS)
            print(f"Created alias: {REPORT_PDF_ALIAS.name}")
            
    # 2. Generate Technical Documentation Manual PDF
    if DOCS_MD.exists():
        docs_text = DOCS_MD.read_text(encoding="utf-8")
        docs_html = markdown_to_html(
            docs_text,
            title="CSCD608: Advanced Computer Vision — Technical Documentation & User Manual",
            subtitle="Complete A-to-Z Technical Reference, Geometry Formulations, and API Guide",
            doc_type="docs"
        )
        docs_html_path = ROOT_DIR / "docs" / "docs_print.html"
        docs_html_path.write_text(docs_html, encoding="utf-8")
        
        convert_html_to_pdf(docs_html_path, DOCS_PDF)
        if DOCS_PDF.exists():
            import shutil
            shutil.copyfile(DOCS_PDF, DOCS_PDF_ALIAS)
            print(f"Created alias: {DOCS_PDF_ALIAS.name}")

    print("\n" + "="*70)
    print("ALL SUBMISSION PDFS GENERATED SUCCESSFULLY!")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
