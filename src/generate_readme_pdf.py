from __future__ import annotations

import re
import textwrap
from pathlib import Path
from xml.sax.saxutils import escape

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Paragraph, Preformatted, SimpleDocTemplate, Spacer

from project_paths import DOCS_DIR, ROOT, ensure_output_dirs

README_PATH = ROOT.parent / "README.md"
OUTPUT_PATH = DOCS_DIR / "README_project.pdf"

PAGE_SIZE = letter
LEFT_MARGIN = 0.8 * inch
RIGHT_MARGIN = 0.8 * inch
TOP_MARGIN = 0.75 * inch
BOTTOM_MARGIN = 0.7 * inch
BODY_FONT_SIZE = 10
CODE_FONT_SIZE = 8.6
MAX_CODE_CHARS = 92


def register_fonts() -> tuple[str, str]:
    code_candidates = [
        (Path("C:/Windows/Fonts/consola.ttf"), "Consolas"),
        (Path("C:/Windows/Fonts/DejaVuSansMono.ttf"), "DejaVuSansMono"),
    ]
    body_candidates = [
        (Path("C:/Windows/Fonts/arial.ttf"), "Arial"),
        (Path("C:/Windows/Fonts/segoeui.ttf"), "SegoeUI"),
    ]

    code_font = "Courier"
    body_font = "Helvetica"

    for path, name in code_candidates:
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


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    return {
        "title": ParagraphStyle(
            "title",
            parent=sample["Title"],
            fontName=BODY_FONT,
            fontSize=22,
            leading=26,
            textColor=colors.HexColor("#0f172a"),
            spaceAfter=10,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            parent=sample["Normal"],
            fontName=BODY_FONT,
            fontSize=11,
            leading=14,
            textColor=colors.HexColor("#475569"),
            spaceAfter=14,
        ),
        "section": ParagraphStyle(
            "section",
            parent=sample["Heading2"],
            fontName=BODY_FONT,
            fontSize=14,
            leading=18,
            textColor=colors.HexColor("#111827"),
            spaceBefore=10,
            spaceAfter=6,
        ),
        "subsection": ParagraphStyle(
            "subsection",
            parent=sample["Heading3"],
            fontName=BODY_FONT,
            fontSize=11.5,
            leading=14,
            textColor=colors.HexColor("#1f2937"),
            spaceBefore=8,
            spaceAfter=5,
        ),
        "body": ParagraphStyle(
            "body",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=BODY_FONT_SIZE,
            leading=13.5,
            textColor=colors.HexColor("#111827"),
            spaceAfter=7,
        ),
        "bullet": ParagraphStyle(
            "bullet",
            parent=sample["BodyText"],
            fontName=BODY_FONT,
            fontSize=BODY_FONT_SIZE,
            leading=13.5,
            textColor=colors.HexColor("#111827"),
            leftIndent=14,
            firstLineIndent=-10,
            spaceAfter=5,
        ),
        "code": ParagraphStyle(
            "code",
            parent=sample["Code"],
            fontName=CODE_FONT,
            fontSize=CODE_FONT_SIZE,
            leading=10.2,
            textColor=colors.black,
            leftIndent=8,
            rightIndent=4,
            borderPadding=7,
            backColor=colors.HexColor("#f8fafc"),
            borderColor=colors.HexColor("#d1d5db"),
            borderWidth=0.6,
            borderRadius=3,
            spaceAfter=10,
        ),
    }


STYLES = build_styles()


def normalize_text(text: str) -> str:
    return text.replace("\t", "    ").replace("\r\n", "\n").replace("\r", "\n")


def replace_markdown_links(text: str) -> str:
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1 (\2)", text)


def format_inline_markdown(text: str) -> str:
    text = replace_markdown_links(text)
    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(
        r"`([^`]+)`",
        lambda match: f'<font name="{CODE_FONT}">{match.group(1)}</font>',
        text,
    )
    return text


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
    wrapped_lines: list[str] = []
    for raw_line in normalize_text(text).split("\n"):
        wrapped_lines.extend(wrap_code_line(raw_line, width=width))
    return "\n".join(wrapped_lines)


def draw_page_header_footer(canvas, doc) -> None:
    canvas.saveState()
    canvas.setFont(BODY_FONT, 9)
    canvas.setFillColor(colors.HexColor("#475569"))
    canvas.drawString(LEFT_MARGIN, PAGE_SIZE[1] - 0.35 * inch, "ShelbyFirst README")
    canvas.drawRightString(PAGE_SIZE[0] - RIGHT_MARGIN, 0.32 * inch, f"Page {doc.page}")
    canvas.restoreState()


def build_story(readme_path: Path) -> list:
    text = normalize_text(readme_path.read_text(encoding="utf-8"))
    lines = text.split("\n")

    story: list = []
    paragraph_buffer: list[str] = []
    code_buffer: list[str] = []
    in_code_block = False
    title_added = False

    def flush_paragraph() -> None:
        if not paragraph_buffer:
            return
        paragraph_text = " ".join(part.strip() for part in paragraph_buffer if part.strip())
        if paragraph_text:
            story.append(Paragraph(format_inline_markdown(paragraph_text), STYLES["body"]))
        paragraph_buffer.clear()

    def flush_code_block() -> None:
        nonlocal in_code_block
        if code_buffer:
            story.append(Preformatted(wrap_code_block("\n".join(code_buffer)), STYLES["code"]))
            code_buffer.clear()
        in_code_block = False

    for raw_line in lines:
        line = raw_line.rstrip()
        stripped = line.strip()

        if stripped.startswith("```"):
            flush_paragraph()
            if in_code_block:
                flush_code_block()
            else:
                in_code_block = True
            continue

        if in_code_block:
            code_buffer.append(line)
            continue

        if not stripped:
            flush_paragraph()
            continue

        if stripped.startswith("# "):
            flush_paragraph()
            title_text = stripped[2:].strip()
            story.append(Paragraph(format_inline_markdown(title_text), STYLES["title"]))
            story.append(
                Paragraph(
                    "Portfolio summary, analytical findings, repository structure, and reproduction notes.",
                    STYLES["subtitle"],
                )
            )
            title_added = True
            continue

        if stripped.startswith("## "):
            flush_paragraph()
            story.append(Paragraph(format_inline_markdown(stripped[3:].strip()), STYLES["section"]))
            continue

        if stripped.startswith("### "):
            flush_paragraph()
            story.append(Paragraph(format_inline_markdown(stripped[4:].strip()), STYLES["subsection"]))
            continue

        if stripped.startswith("- "):
            flush_paragraph()
            bullet_text = "&bull; " + format_inline_markdown(stripped[2:].strip())
            story.append(Paragraph(bullet_text, STYLES["bullet"]))
            continue

        if re.match(r"\d+\.\s+", stripped):
            flush_paragraph()
            marker, text_part = stripped.split(" ", 1)
            story.append(Paragraph(f"{marker} {format_inline_markdown(text_part)}", STYLES["bullet"]))
            continue

        paragraph_buffer.append(stripped)

    flush_paragraph()
    if in_code_block:
        flush_code_block()

    if not title_added:
        story.insert(0, Paragraph("Repository README", STYLES["title"]))

    story.append(Spacer(1, 0.1 * inch))
    return story


def build_pdf(readme_path: Path, output_path: Path) -> None:
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=PAGE_SIZE,
        leftMargin=LEFT_MARGIN,
        rightMargin=RIGHT_MARGIN,
        topMargin=TOP_MARGIN,
        bottomMargin=BOTTOM_MARGIN,
        title="ShelbyFirst README",
        author="Codex",
    )

    story = build_story(readme_path)
    doc.build(story, onFirstPage=draw_page_header_footer, onLaterPages=draw_page_header_footer)


def main() -> None:
    if not README_PATH.exists():
        raise FileNotFoundError(f"Missing README file: {README_PATH}")

    ensure_output_dirs()
    build_pdf(README_PATH, OUTPUT_PATH)
    print(f"Generated: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
