"""Batch processing for the line grid intersection pipeline.

Usage:
    python -m ex_intersect.batch_count_intersects_line_grid \
        --input-dir /path/to/images --output-dir /path/to/results [options]

Supported image formats: PNG, JPG, JPEG, TIFF, TIF, BMP.
"""

from __future__ import annotations

import argparse
import traceback
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import pandas as pd

from ex_intersect import line_grid_pipeline as pipeline
from ex_intersect.count_intersects_line_grid import configure_plot_style

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def build_parser() -> argparse.ArgumentParser:
    """Create the command line parser for the batch runner."""

    parser = argparse.ArgumentParser(
        description=(
            "Batch process segmented micrographs with the line-grid intersection "
            "pipeline."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help="Directory containing segmented binary images that should be analysed.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory where result artefacts for all images will be stored.",
    )
    parser.add_argument(
        "--summary-excel",
        type=Path,
        default=None,
        help=(
            "Optional path to the Excel workbook collecting summary rows across "
            "all processed images. Defaults to '<output-dir>/summary.xlsx'."
        ),
    )

    parser.add_argument(
        "--row-step",
        type=int,
        default=20,
        help="Row sampling step that controls how densely the grid is sampled.",
    )
    parser.add_argument(
        "--theta-start",
        type=float,
        default=0.0,
        help="First rotation angle of the virtual line grid in degrees.",
    )
    parser.add_argument(
        "--theta-end",
        type=float,
        default=180.0,
        help="Last rotation angle of the virtual line grid in degrees.",
    )
    parser.add_argument(
        "--theta-steps",
        type=int,
        default=6,
        help="Number of rotation angles that should be evaluated.",
    )
    parser.add_argument(
        "--inclusive-theta-end",
        action="store_true",
        help="Include the end angle as part of the rotation sweep.",
    )

    parser.add_argument(
        "--scalebar-pixel",
        type=float,
        default=464.0,
        help="Pixel length of the scale bar drawn on the image.",
    )
    parser.add_argument(
        "--scalebar-micrometer",
        type=float,
        default=50.0,
        help="Physical length of the scale bar in micrometres.",
    )
    parser.add_argument(
        "--crop-rows",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=(0, 1825),
        help="Row range that should be analysed (start inclusive, end exclusive).",
    )
    parser.add_argument(
        "--crop-cols",
        type=int,
        nargs=2,
        metavar=("START", "END"),
        default=(0, 2580),
        help="Column range that should be analysed (start inclusive, end exclusive).",
    )

    parser.add_argument(
        "--borders-black",
        dest="borders_white",
        action="store_false",
        help="Invert the binary image before processing because borders are dark.",
    )
    parser.add_argument(
        "--no-reskeletonize",
        dest="reskeletonize",
        action="store_false",
        help="Skip morphological clean-up before measuring intersections.",
    )
    parser.set_defaults(borders_white=True, reskeletonize=True)

    parser.add_argument(
        "--save-rotated-images",
        action="store_true",
        help="Persist each rotated intermediate image used during processing.",
    )
    parser.add_argument(
        "--no-boxplot",
        dest="save_boxplot",
        action="store_false",
        help="Disable creation of the distribution boxplot.",
    )
    parser.add_argument(
        "--no-histograms",
        dest="save_histograms",
        action="store_false",
        help="Disable creation of histogram graphics for the distances.",
    )
    parser.add_argument(
        "--no-excel",
        dest="save_excel",
        action="store_false",
        help="Skip writing the detailed Excel workbook with the results.",
    )
    parser.add_argument(
        "--no-summary",
        dest="append_summary",
        action="store_false",
        help="Do not append a summary row to the shared summary workbook.",
    )
    parser.add_argument(
        "--show-plots",
        action="store_true",
        help="Show Matplotlib figures instead of closing them automatically.",
    )
    parser.set_defaults(
        save_boxplot=True,
        save_histograms=True,
        save_excel=True,
        append_summary=True,
        show_plots=False,
    )

    return parser


def find_image_files(input_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """Return a sorted list of image files inside ``input_dir`` with given suffixes."""

    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files)


def build_config_and_options(
    image_path: Path, args: argparse.Namespace, results_root: Path, summary_path: Path
) -> Tuple[pipeline.LineGridConfig, pipeline.SaveOptions]:
    """Create configuration objects for the current image."""

    config = pipeline.LineGridConfig(
        file_in_path=image_path,
        results_base_dir=results_root,
        summary_excel_path=summary_path,
        borders_white=args.borders_white,
        row_step=args.row_step,
        theta_start=args.theta_start,
        theta_end=args.theta_end,
        n_theta_steps=args.theta_steps,
        inclusive_theta_end=args.inclusive_theta_end,
        reskeletonize=args.reskeletonize,
        scalebar_pixel=args.scalebar_pixel,
        scalebar_micrometer=args.scalebar_micrometer,
        crop_rows=tuple(args.crop_rows),
        crop_cols=tuple(args.crop_cols),
    )

    options = pipeline.SaveOptions(
        save_rotated_images=args.save_rotated_images,
        save_boxplot=args.save_boxplot,
        save_histograms=args.save_histograms,
        save_excel=args.save_excel,
        append_summary=args.append_summary,
        show_plots=args.show_plots,
    )

    return config, options


def update_summary_aggregates(summary_path: Path) -> Optional[pd.DataFrame]:
    """Recompute aggregate statistics for the summary workbook."""

    if not summary_path.exists():
        return None

    try:
        summary_df = pd.read_excel(summary_path)
    except Exception as exc:  # pragma: no cover - diagnostic output
        print(f"[WARNING] Could not read summary workbook '{summary_path}': {exc}")
        return None

    if summary_df.empty:
        return None

    numeric_df = summary_df.select_dtypes(include=["number"])
    if numeric_df.empty:
        return None

    aggregates = numeric_df.agg(["count", "mean", "median"])
    aggregates = aggregates.reset_index().rename(columns={"index": "Metric"})

    try:
        with pd.ExcelWriter(
            summary_path,
            mode="a",
            engine="openpyxl",
            if_sheet_exists="replace",
        ) as writer:
            aggregates.to_excel(writer, sheet_name="Aggregates", index=False)
    except Exception as exc:  # pragma: no cover - diagnostic output
        print(
            f"[WARNING] Failed to update aggregate statistics in '{summary_path}': {exc}"
        )
        return None

    return aggregates


def ensure_directory(path: Path) -> None:
    """Create ``path`` if it does not already exist."""

    path.mkdir(parents=True, exist_ok=True)


def process_images(args: argparse.Namespace) -> None:
    """Iterate over all images and execute the processing pipeline."""

    input_dir: Path = args.input_dir
    output_dir: Path = args.output_dir

    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory '{input_dir}' does not exist.")

    ensure_directory(output_dir)

    summary_path = args.summary_excel or (output_dir / "summary.xlsx")

    image_files = find_image_files(input_dir, SUPPORTED_EXTENSIONS)
    if not image_files:
        print(
            f"[INFO] No image files with supported extensions {SUPPORTED_EXTENSIONS} "
            f"found in '{input_dir}'."
        )
        return

    failures: List[Tuple[Path, Exception]] = []

    print(f"[INFO] Found {len(image_files)} image(s) to process.")

    for index, image_path in enumerate(image_files, start=1):
        rel_parent = image_path.parent.relative_to(input_dir)
        results_root = output_dir / rel_parent
        ensure_directory(results_root)

        print(f"[INFO] Processing {image_path} ({index}/{len(image_files)})")

        config, options = build_config_and_options(
            image_path=image_path,
            args=args,
            results_root=results_root,
            summary_path=summary_path,
        )

        try:
            statistics, artifacts = pipeline.process_image(config, options)
        except Exception as exc:  # pragma: no cover - runtime diagnostic
            print(f"[ERROR] Failed to process '{image_path}': {exc}")
            traceback.print_exc()
            failures.append((image_path, exc))
            continue

        excel_target = artifacts.excel_path or config.results_base_dir
        overall_stats = statistics.overall_statistics
        print(
            f"[INFO] Finished processing '{image_path.name}'. Results stored under "
            f"'{excel_target}'."
        )
        print(
            "[INFO] Overall average grain size: "
            f"{overall_stats.average_length:.2f} µm (median: {overall_stats.median_length:.2f} µm)"
        )

        if artifacts.summary_excel_path:
            print(
                f"[INFO] Summary workbook updated: {artifacts.summary_excel_path}"
            )

        if options.append_summary:
            aggregates = update_summary_aggregates(summary_path)
            if aggregates is not None:
                print(
                    "[INFO] Updated aggregate statistics in the summary workbook: "
                    f"{summary_path}"
                )

    if failures:
        print("\n[SUMMARY] Processing completed with errors:")
        for failed_path, exc in failures:
            print(f"  - {failed_path}: {exc}")
    else:
        print("\n[SUMMARY] All images processed successfully.")


def main(args: Optional[Sequence[str]] = None) -> None:
    """Parse arguments, configure Matplotlib, and execute the batch run."""

    parser = build_parser()
    parsed_args = parser.parse_args(args)
    configure_plot_style()
    process_images(parsed_args)


if __name__ == "__main__":
    main()
