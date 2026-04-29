from __future__ import annotations

import json
import math
import re
import textwrap
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.pagesizes import landscape, letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import PageBreak, Paragraph, Preformatted, SimpleDocTemplate, Spacer

from project_paths import MANIFESTS_DIR, ROOT, SOURCE_PDFS_DIR, ensure_output_dirs

SOURCE_FILES = [
    ("final_model.ipynb", "final_model_source.pdf", "Final Model"),
    ("final_model2.ipynb", "final_model2_source.pdf", "Final Model 2"),
    ("shelby_map_black_pct.py", "shelby_map_black_pct_source.pdf", "Shelby Map Black Pct"),
    ("shelbyfirst_outcomes_chart.py", "shelbyfirst_outcomes_chart_source.pdf", "ShelbyFirst Outcomes Chart"),
    ("shelbyfirst_simulation.ipynb", "shelbyfirst_simulation_source.pdf", "ShelbyFirst Simulation"),
    ("shelbyfirst_table_chart.py", "shelbyfirst_table_chart_source.pdf", "ShelbyFirst Table Chart"),
]

MASTER_PDF_NAME = "submission_source_bundle.pdf"

PAGE_SIZE = landscape(letter)
LEFT_MARGIN = 0.5 * inch
RIGHT_MARGIN = 0.5 * inch
TOP_MARGIN = 0.6 * inch
BOTTOM_MARGIN = 0.5 * inch
CODE_FONT_SIZE = 7.2
BODY_FONT_SIZE = 9.0
TITLE_FONT_SIZE = 20
SUBTITLE_FONT_SIZE = 11
MAX_CODE_CHARS = 132


def register_fonts() -> tuple[str, str]:
    font_candidates = [
        (Path("C:/Windows/Fonts/consola.ttf"), "Consolas"),
        (Path("C:/Windows/Fonts/DejaVuSansMono.ttf"), "DejaVuSansMono"),
    ]
    body_candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), "Arial"),
        (Path("C:/Windows/Fonts/segoeui.ttf"), "SegoeUI"),
    ]

    code_font = "Courier"
    body_font = "Helvetica"

    for path, name in font_candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
            code_font = name
            break

    for path, name in body_candidates:
        if path.exists():
            pdfmetrics.registerFont(TTFont(name, str(path)))
            body_font = name
            break

    return code_font, body_font


CODE_FONT, BODY_FONT = register_fonts()


def sanitize_text(text: str) -> str:
    return text.replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n")


def line_indent(line: str) -> str:
    match = re.match(r"\s*", line)
    return match.group(0) if match else ""


def wrap_code_line(line: str, width: int = MAX_CODE_CHARS) -> list[str]:
    if not line:
        return [""]
    if len(line) <= width:
        return [line]

    indent = line_indent(line)
    body = line[len(indent):]
    wrapped = textwrap.wrap(
        body,
        width=max(width - len(indent), 20),
        break_long_words=False,
        break_on_hyphens=False,
        initial_indent=indent,
        subsequent_indent=indent + "    ",
        replace_whitespace=False,
        drop_whitespace=False,
    )
    return wrapped if wrapped else [line]


def wrap_code_block(text: str, width: int = MAX_CODE_CHARS) -> str:
    lines = []
    for raw_line in sanitize_text(text).split("\n"):
        lines.extend(wrap_code_line(raw_line, width=width))
    return "\n".join(lines)


def load_notebook_cells(path: Path) -> list[dict]:
    nb = json.loads(path.read_text(encoding="utf-8"))
    return nb.get("cells", [])


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    styles = {
        "title": ParagraphStyle(
            "title",
            parent=sample["Title"],
            fontName=BODY_FONT,
            fontSize=TITLE_FONT_SIZE,
            leading=24,
            textColor=colors.HexColor("#1f2937"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=sample["Normal"],
            fontName=BODY_FONT,
            fontSize=SUBTITLE_FONT_SIZE,
            leading=14,
            textColor=colors.HexColor("#4b5563"),
            spaceAfter=14,
        ),
        "heading": ParagraphStyle(
            "heading",
            parent=sample["Heading2"],
            fontName=BODY_FONT,
            fontSize=12,
            leading=15,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
            spaceBefore=8,
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample["Normal"],
            fontName=BODY_FONT,
            fontSize=BODY_FONT_SIZE,
            leading=12,
            textColor=colors.HexColor("#111827"),
            spaceAfter=6,
        ),
        "code": ParagraphStyle(
            "code",
            parent=sample["Code"],
            fontName=CODE_FONT,
            fontSize=CODE_FONT_SIZE,
            leading=8.8,
            textColor=colors.black,
            leftIndent=8,
            rightIndent=4,
            borderPadding=6,
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#d1d5db"),
            borderWidth=0.5,
            borderRadius=3,
            spaceAfter=8,
        ),
    }
    return styles


STYLES = build_styles()


def draw_page_header_footer(canvas, doc, source_label: str) -> None:
    canvas.saveState()
    canvas.setFont(BODY_FONT, 9)
    canvas.setFillColor(colors.HexColor("#4b5563"))
    canvas.drawString(LEFT_MARGIN, PAGE_SIZE[1] - 0.35 * inch, source_label)
    canvas.drawRightString(PAGE_SIZE[0] - RIGHT_MARGIN, 0.3 * inch, f"Page {doc.page}")
    canvas.restoreState()


def notebook_story(path: Path, display_name: str) -> list:
    story = [
        Paragraph(display_name, STYLES["title"]),
        Paragraph(f"Notebook source rendered from {path.name} (inputs only, outputs omitted).", STYLES["subtitle"]),
    ]

    cells = load_notebook_cells(path)
    story.append(Paragraph(f"Total cells: {len(cells)}", STYLES["body"]))
    story.append(Spacer(1, 0.12 * inch))

    for idx, cell in enumerate(cells, start=1):
        cell_type = str(cell.get("cell_type", "unknown")).title()
        source = cell.get("source", "")
        if isinstance(source, list):
            source = "".join(source)
        source = sanitize_text(source)
        if not source.strip():
            continue

        story.append(Paragraph(f"Cell {idx} — {cell_type}", STYLES["heading"]))

        if cell_type.lower() == "markdown":
            story.append(Preformatted(wrap_code_block(source), STYLES["code"]))
        else:
            story.append(Preformatted(wrap_code_block(source), STYLES["code"]))

    return story


def python_story(path: Path, display_name: str) -> list:
    source = sanitize_text(path.read_text(encoding="utf-8"))
    line_count = source.count("\n") + 1

    return [
        Paragraph(display_name, STYLES["title"]),
        Paragraph(f"Python source rendered from {path.name}.", STYLES["subtitle"]),
        Paragraph(f"Approximate line count: {line_count}", STYLES["body"]),
        Spacer(1, 0.12 * inch),
        Preformatted(wrap_code_block(source), STYLES["code"]),
    ]


def build_pdf(source_path: Path, output_pdf: Path, display_name: str) -> None:
    doc = SimpleDocTemplate(
        str(output_pdf),
        pagesize=PAGE_SIZE,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title=display_name,
        author="Codex",
    )

    if source_path.suffix.lower() == ".ipynb":
        story = notebook_story(source_path, display_name)
    else:
        story = python_story(source_path, display_name)

    def add_header_footer(canvas, doc_obj):
        draw_page_header_footer(canvas, doc_obj, display_name)

    doc.build(story, onFirstPage=add_header_footer, onLaterPages=add_header_footer)


def merge_pdfs(pdf_paths: list[Path], merged_path: Path) -> None:
    writer = PdfWriter()
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)
    with merged_path.open("wb") as f:
        writer.write(f)


def write_manifest(rows: list[tuple[str, str, str]], manifest_path: Path) -> None:
    lines = ["Submission PDF package", ""]
    for source_name, pdf_name, display_name in rows:
        lines.append(f"- {display_name}: {source_name} -> {pdf_name}")
    manifest_path.write_text("\n".join(lines), encoding="utf-8")


def build_submission_source_pdfs(output_dir: Path | None = None) -> list[Path]:
    ensure_output_dirs()
    output_dir = output_dir or SOURCE_PDFS_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    generated_pdfs: list[Path] = []
    for source_name, pdf_name, display_name in SOURCE_FILES:
        source_path = ROOT / source_name
        if not source_path.exists():
            raise FileNotFoundError(f"Missing source file: {source_path}")
        output_pdf = output_dir / pdf_name
        build_pdf(source_path, output_pdf, display_name)
        generated_pdfs.append(output_pdf)
        print(f"Generated: {output_pdf}")

    merged_path = output_dir / MASTER_PDF_NAME
    merge_pdfs(generated_pdfs, merged_path)
    print(f"Generated merged bundle: {merged_path}")

    manifest_path = MANIFESTS_DIR / "source_pdf_manifest.txt"
    write_manifest(SOURCE_FILES, manifest_path)
    print(f"Generated manifest: {manifest_path}")

    return generated_pdfs + [merged_path]


def main() -> None:
    build_submission_source_pdfs()


if __name__ == "__main__":
    main()
