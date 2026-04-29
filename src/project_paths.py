from __future__ import annotations

import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parent

DATA_DIR = ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PRESENTATION_READY_DIR = PROCESSED_DATA_DIR / "presentation_ready"

OUTPUTS_DIR = ROOT / "outputs"
SUBMISSION_PACKAGE_DIR = OUTPUTS_DIR / "submission_package"

FIGURES_DIR = SUBMISSION_PACKAGE_DIR / "figures"
MAPS_DIR = FIGURES_DIR / "maps"
CHARTS_DIR = FIGURES_DIR / "charts"
PRESENTATION_ASSETS_DIR = FIGURES_DIR / "presentation_assets"
PRESENTATION_READY_ASSETS_DIR = PRESENTATION_ASSETS_DIR / "presentation_ready"
PROCESSED_FIGURE_ASSETS_DIR = PRESENTATION_ASSETS_DIR / "processed"

DOCS_DIR = SUBMISSION_PACKAGE_DIR / "docs"
PDFS_DIR = SUBMISSION_PACKAGE_DIR / "pdfs"
SOURCE_PDFS_DIR = PDFS_DIR / "source"
FIGURE_PDFS_DIR = PDFS_DIR / "figures"
HTML_DIR = SUBMISSION_PACKAGE_DIR / "html"
MANIFESTS_DIR = SUBMISSION_PACKAGE_DIR / "manifests"


def ensure_output_dirs() -> None:
    for path in [
        OUTPUTS_DIR,
        SUBMISSION_PACKAGE_DIR,
        FIGURES_DIR,
        MAPS_DIR,
        CHARTS_DIR,
        PRESENTATION_ASSETS_DIR,
        PRESENTATION_READY_ASSETS_DIR,
        PROCESSED_FIGURE_ASSETS_DIR,
        DOCS_DIR,
        PDFS_DIR,
        SOURCE_PDFS_DIR,
        FIGURE_PDFS_DIR,
        HTML_DIR,
        MANIFESTS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)
