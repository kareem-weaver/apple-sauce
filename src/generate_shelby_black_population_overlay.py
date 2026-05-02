from __future__ import annotations

import argparse
from pathlib import Path

from generate_shelby_slidedeck_visuals import OUTPUT_FILES, generate_black_population_map
from project_paths import PRESENTATION_READY_OUTPUT_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate the Shelby slide-deck Black population map."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PRESENTATION_READY_OUTPUT_DIR / OUTPUT_FILES["black-map"],
        help="Output PNG path. Defaults to src/outputs/presentation_ready/.",
    )
    return parser.parse_args()


def main(output_path: Path | None = None) -> Path:
    target = output_path or (PRESENTATION_READY_OUTPUT_DIR / OUTPUT_FILES["black-map"])
    return generate_black_population_map(target)


if __name__ == "__main__":
    args = parse_args()
    print(main(output_path=args.output))
