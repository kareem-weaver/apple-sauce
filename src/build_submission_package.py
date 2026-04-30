from __future__ import annotations

import json
import shutil
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from pypdf import PdfReader, PdfWriter
from reportlab.lib.pagesizes import letter, landscape
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas

from project_paths import (
    CHARTS_DIR,
    DOCS_DIR,
    FIGURE_PDFS_DIR,
    FIGURES_DIR,
    HTML_DIR,
    MANIFESTS_DIR,
    MAPS_DIR,
    PRESENTATION_READY_ASSETS_DIR,
    PRESENTATION_READY_DIR,
    PROCESSED_DATA_DIR,
    PROCESSED_FIGURE_ASSETS_DIR,
    ROOT,
    SOURCE_PDFS_DIR,
    SUBMISSION_PACKAGE_DIR,
    ensure_output_dirs,
    reset_directory,
)


CORE_FIGURE_OUTPUTS = {
    "maps": [
        MAPS_DIR / "shelby_black_share_choropleth.png",
        MAPS_DIR / "shelby_internet_access_choropleth.png",
        MAPS_DIR / "shelby_map_black_pct.png",
    ],
    "charts": [
        CHARTS_DIR / "shelby_internet_access_bar_chart.png",
        CHARTS_DIR / "shelby_no_internet_bar_chart.png",
        CHARTS_DIR / "shelbyfirst_outcomes_chart.png",
        CHARTS_DIR / "shelbyfirst_table_chart.png",
    ],
}

CORE_FIGURE_NAMES = {path.name for paths in CORE_FIGURE_OUTPUTS.values() for path in paths}

SCRIPT_STEPS = [
    (
        "Shelby Black Share Choropleth",
        [sys.executable, "generate_shelby_black_population_overlay.py", "--output", str(CORE_FIGURE_OUTPUTS["maps"][0])],
    ),
    (
        "Shelby Internet Access Choropleth",
        [sys.executable, "generate_shelby_internet_access_overlay.py", "--output", str(CORE_FIGURE_OUTPUTS["maps"][1])],
    ),
    (
        "Shelby Comparator Map",
        [sys.executable, "shelby_map_black_pct.py", "--output", str(CORE_FIGURE_OUTPUTS["maps"][2])],
    ),
    (
        "Shelby Internet Access Bar Chart",
        [sys.executable, "shelby_internet_access_bar_chart.py", "--output", str(CORE_FIGURE_OUTPUTS["charts"][0])],
    ),
    (
        "Shelby No-Internet Bar Chart",
        [
            sys.executable,
            "shelby_internet_access_bar_chart.py",
            "--metric",
            "no_internet",
            "--output",
            str(CORE_FIGURE_OUTPUTS["charts"][1]),
        ],
    ),
    (
        "ShelbyFirst Outcomes Chart",
        [sys.executable, "shelbyfirst_outcomes_chart.py"],
    ),
    (
        "ShelbyFirst Table Chart",
        [sys.executable, "shelbyfirst_table_chart.py"],
    ),
    (
        "README PDF",
        [sys.executable, "generate_readme_pdf.py"],
    ),
    (
        "Source PDFs",
        [sys.executable, "generate_submission_source_pdfs.py"],
    ),
]

HTML_GLOBS = [
    "shelbyfirst_*simulation.html",
    "shelbyfirst_summary_chart.html",
]


def run_step(label: str, command: list[str]) -> None:
    print(f"[run] {label}")
    subprocess.run(command, cwd=ROOT, check=True)


def copy_pngs(source_dir: Path, target_dir: Path, exclude_names: set[str] | None = None) -> list[Path]:
    copied: list[Path] = []
    exclude_names = exclude_names or set()
    target_dir.mkdir(parents=True, exist_ok=True)

    for png_path in sorted(source_dir.glob("*.png")):
        if png_path.name in exclude_names:
            continue
        destination = target_dir / png_path.name
        shutil.copy2(png_path, destination)
        copied.append(destination)

    return copied


def copy_html_assets() -> list[Path]:
    copied: list[Path] = []
    for pattern in HTML_GLOBS:
        for html_path in sorted(PROCESSED_DATA_DIR.glob(pattern)):
            destination = HTML_DIR / html_path.name
            shutil.copy2(html_path, destination)
            copied.append(destination)
    return copied


def render_png_pdf(image_path: Path, output_pdf: Path) -> None:
    output_pdf.parent.mkdir(parents=True, exist_ok=True)
    image = ImageReader(str(image_path))
    img_width, img_height = image.getSize()

    if img_width >= img_height:
        page_width, page_height = landscape(letter)
    else:
        page_width, page_height = letter

    margin = 0.45 * inch
    usable_width = page_width - 2 * margin
    usable_height = page_height - 2 * margin
    scale = min(usable_width / img_width, usable_height / img_height)
    draw_width = img_width * scale
    draw_height = img_height * scale
    x = (page_width - draw_width) / 2
    y = (page_height - draw_height) / 2

    pdf = canvas.Canvas(str(output_pdf), pagesize=(page_width, page_height))
    pdf.setTitle(image_path.stem)
    pdf.drawImage(image, x, y, width=draw_width, height=draw_height, preserveAspectRatio=True, mask="auto")
    pdf.showPage()
    pdf.save()


def build_figure_pdfs() -> list[Path]:
    generated: list[Path] = []
    for png_path in sorted(FIGURES_DIR.rglob("*.png")):
        relative_pdf = png_path.relative_to(FIGURES_DIR).with_suffix(".pdf")
        output_pdf = FIGURE_PDFS_DIR / relative_pdf
        render_png_pdf(png_path, output_pdf)
        generated.append(output_pdf)
        print(f"[pdf] {output_pdf}")
    return generated


def merge_pdfs(pdf_paths: list[Path], output_path: Path) -> Path:
    writer = PdfWriter()
    for pdf_path in pdf_paths:
        reader = PdfReader(str(pdf_path))
        for page in reader.pages:
            writer.add_page(page)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as handle:
        writer.write(handle)

    return output_path


def write_manifest(manifest: dict) -> None:
    json_path = MANIFESTS_DIR / "submission_package_manifest.json"
    txt_path = MANIFESTS_DIR / "submission_package_manifest.txt"

    json_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lines = [
        "Submission package manifest",
        "",
        f"Generated at: {manifest['generated_at']}",
        f"Package root: {manifest['package_root']}",
        "",
        "Counts:",
    ]
    for key, value in manifest["counts"].items():
        lines.append(f"- {key}: {value}")

    lines.extend(
        [
            "",
            "Core maps:",
            *[f"- {path}" for path in manifest["core_maps"]],
            "",
            "Core charts:",
            *[f"- {path}" for path in manifest["core_charts"]],
            "",
            "Docs:",
            *[f"- {path}" for path in manifest["docs"]],
            "",
            "Source PDFs:",
            *[f"- {path}" for path in manifest["source_pdfs"]],
            "",
            "Figure bundles:",
            *[f"- {path}" for path in manifest["bundles"]],
        ]
    )

    txt_path.write_text("\n".join(lines), encoding="utf-8")


def rel_paths(paths: list[Path]) -> list[str]:
    return [str(path.relative_to(ROOT)) for path in sorted(paths)]


def build_submission_package() -> dict:
    reset_directory(SUBMISSION_PACKAGE_DIR)
    ensure_output_dirs()

    for label, command in SCRIPT_STEPS:
        run_step(label, command)

    copied_presentation = copy_pngs(
        PRESENTATION_READY_DIR,
        PRESENTATION_READY_ASSETS_DIR,
        exclude_names=CORE_FIGURE_NAMES,
    )
    copied_processed = copy_pngs(
        PROCESSED_DATA_DIR,
        PROCESSED_FIGURE_ASSETS_DIR,
        exclude_names=CORE_FIGURE_NAMES,
    )
    copied_html = copy_html_assets()

    figure_pdfs = build_figure_pdfs()
    figure_bundle = merge_pdfs(figure_pdfs, FIGURE_PDFS_DIR / "submission_figure_bundle.pdf")

    docs = sorted(DOCS_DIR.glob("*.pdf"))
    source_pdfs = sorted(SOURCE_PDFS_DIR.glob("*.pdf"))
    map_pngs = sorted(MAPS_DIR.glob("*.png"))
    chart_pngs = sorted(CHARTS_DIR.glob("*.png"))

    manifest = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "package_root": str(SUBMISSION_PACKAGE_DIR.relative_to(ROOT)),
        "counts": {
            "core_maps": len(map_pngs),
            "core_charts": len(chart_pngs),
            "presentation_ready_assets": len(copied_presentation),
            "processed_assets": len(copied_processed),
            "html_assets": len(copied_html),
            "figure_pdfs": len(figure_pdfs),
            "source_pdfs": len(source_pdfs),
            "doc_pdfs": len(docs),
        },
        "core_maps": rel_paths(map_pngs),
        "core_charts": rel_paths(chart_pngs),
        "docs": rel_paths(docs),
        "source_pdfs": rel_paths(source_pdfs),
        "figure_pdfs": rel_paths(figure_pdfs),
        "html_assets": rel_paths(copied_html),
        "bundles": rel_paths([figure_bundle]),
    }
    write_manifest(manifest)
    return manifest


def main() -> None:
    manifest = build_submission_package()
    print("[done] submission package built")
    for key, value in manifest["counts"].items():
        print(f"  {key}: {value}")
    print(f"  package_root: {manifest['package_root']}")


if __name__ == "__main__":
    main()
