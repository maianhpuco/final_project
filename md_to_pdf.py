#!/usr/bin/env python3
"""
Convert markdown file to PDF using markdown and weasyprint
"""
import sys
import markdown
from weasyprint import HTML, CSS
from pathlib import Path

def markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF"""
    # Get base directory for resolving image paths
    md_path = Path(md_file)
    base_dir = md_path.parent.absolute()
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(
        md_content,
        extensions=['extra', 'codehilite', 'tables', 'fenced_code']
    )
    
    # Add CSS styling for better PDF appearance
    css_style = """
    @page {
        size: A4;
        margin: 2cm;
    }
    body {
        font-family: 'DejaVu Sans', Arial, sans-serif;
        font-size: 11pt;
        line-height: 1.6;
    }
    h1 {
        font-size: 24pt;
        margin-top: 1em;
        margin-bottom: 0.5em;
        page-break-after: avoid;
    }
    h2 {
        font-size: 20pt;
        margin-top: 0.8em;
        margin-bottom: 0.4em;
        page-break-after: avoid;
    }
    h3 {
        font-size: 16pt;
        margin-top: 0.6em;
        margin-bottom: 0.3em;
        page-break-after: avoid;
    }
    code {
        font-family: 'DejaVu Sans Mono', 'Courier New', monospace;
        background-color: #f5f5f5;
        padding: 2px 4px;
        border-radius: 3px;
    }
    pre {
        background-color: #f5f5f5;
        padding: 10px;
        border-radius: 5px;
        overflow-x: auto;
        page-break-inside: avoid;
    }
    pre code {
        background-color: transparent;
        padding: 0;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
        page-break-inside: avoid;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    ul, ol {
        margin: 0.5em 0;
        padding-left: 2em;
    }
    li {
        margin: 0.3em 0;
        line-height: 1.5;
    }
    ul ul, ol ol, ul ol, ol ul {
        margin-top: 0.3em;
        margin-bottom: 0.3em;
    }
    img {
        max-width: 100%;
        max-height: 8cm;
        height: auto;
        width: auto;
        display: block;
        margin: 1em auto;
        page-break-inside: avoid;
        border: 1px solid #ddd;
    }
    p img {
        margin: 1em 0;
    }
    """
    
    # Wrap HTML content with proper structure
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>{css_style}</style>
    </head>
    <body>
    {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF with base_url for image resolution
    HTML(string=full_html, base_url=str(base_dir)).write_pdf(pdf_file)
    print(f"Successfully converted {md_file} to {pdf_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python3 md_to_pdf.py <input.md> [output.pdf]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    if len(sys.argv) > 2:
        pdf_file = sys.argv[2]
    else:
        pdf_file = Path(md_file).with_suffix('.pdf')
    
    if not Path(md_file).exists():
        print(f"Error: File {md_file} not found")
        sys.exit(1)
    
    markdown_to_pdf(md_file, pdf_file)

