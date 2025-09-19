"""Batch processing for the line grid intersection pipeline.

The script reads configuration details from ``batch_count_intersects_line_grid.toml``
located next to this module. Copying that file and adjusting the values allows
you to orchestrate multiple batch runs with different parameters.

Supported image formats: PNG, JPG, JPEG, TIFF, TIF, BMP.
"""

from __future__ import annotations

# --- Import shim: allows "Run Current File" AND `python -m` without breaking package imports ---
import sys
from pathlib import Path as _Path

# project root = parent of package folder that contains this file
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# --- End import shim ---

import traceback
from pathlib import Path
from dataclasses import replace
from typing import List, Optional, Tuple

import pandas as pd

from ex_intersect import line_grid_pipeline as pipeline
from ex_intersect.count_intersects_line_grid import configure_plot_style
from ex_intersect.config_loader import BatchRunConfig, load_batch_run_config

SUPPORTED_EXTENSIONS: Tuple[str, ...] = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_image_files(input_dir: Path, extensions: Tuple[str, ...]) -> List[Path]:
    """Return a sorted list of image files inside ``input_dir``.

    Parameters
    ----------
    input_dir : Path
        Directory that should be searched recursively.
    extensions : tuple of str
        File suffixes (including the leading dot) that should be considered.

    Returns
    -------
    list of Path
        Sorted collection of matching image paths.

    Examples
    --------
    .. code-block:: python

        find_image_files(Path("."), (".nonexistent",))  # -> []
    """
    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files)


def build_config_and_options(
    image_path: Path,
    batch_config: BatchRunConfig,
    results_root: Path,
    summary_path: Path,
) -> Tuple[pipeline.LineGridConfig, pipeline.SaveOptions]:
    """Create configuration objects for the current image.

    Parameters
    ----------
    image_path : Path
        Path to the segmented image that should be processed.
    batch_config : BatchRunConfig
        Parsed configuration loaded from ``batch_count_intersects_line_grid.toml``.
    results_root : Path
        Base directory where artefacts for this image should be stored.
    summary_path : Path
        Location of the shared summary workbook.

    Returns
    -------
    tuple of (:class:`LineGridConfig`, :class:`SaveOptions`)
        Configuration objects used by :func:`pipeline.process_image`.

    Examples
    --------
    .. code-block:: python

        # from ex_intersect.config_loader import BatchRunConfig
        # cfg = BatchRunConfig(Path('.'), Path('./out'), None, {}, pipeline.SaveOptions())
        # config, options = build_config_and_options(Path("image.png"), cfg, Path("."), Path("summary.xlsx"))
    """
    overrides = dict(batch_config.line_grid_overrides)
    overrides.setdefault("results_base_dir", results_root)
    overrides.setdefault("summary_excel_path", summary_path)
    overrides["file_in_path"] = Path(image_path)

    config = pipeline.LineGridConfig(**overrides)
    options = replace(batch_config.save_options)
    return config, options


def update_summary_aggregates(summary_path: Path) -> Optional[pd.DataFrame]:
    """Recompute aggregate statistics for the summary workbook.

    Parameters
    ----------
    summary_path : Path
        Path to the Excel workbook that should receive aggregate statistics.

    Returns
    -------
    pandas.DataFrame or None
        DataFrame with aggregate metrics when successful, otherwise ``None``.

    Examples
    --------
    .. code-block:: python

        # update_summary_aggregates(Path("nonexistent.xlsx"))  # -> None
    """
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
    """Create ``path`` if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory that should be created if missing.

    Examples
    --------
    .. code-block:: python

        ensure_directory(Path("./_tmp_directory"))
        Path("./_tmp_directory").exists()  # -> True
    """
    path.mkdir(parents=True, exist_ok=True)


def process_images(batch_config: BatchRunConfig, *, config_source: Optional[Path] = None) -> None:
    """Iterate over all images and execute the processing pipeline.

    Parameters
    ----------
    batch_config : BatchRunConfig
        Parsed configuration describing the batch run.
    config_source : Path or None, optional
        Path to the TOML file from which ``batch_config`` was loaded. Used for
        informative error messages.

    Raises
    ------
    FileNotFoundError
        If the directory referenced by the ``input_dir`` key does not exist.
    NotADirectoryError
        If ``input_dir`` or ``output_dir`` points to a non-directory path.
    OSError
        If the ``output_dir`` directory cannot be created.
    """
    input_dir = Path(batch_config.input_dir)
    output_dir = Path(batch_config.output_dir)
    config_label = (
        str(config_source) if config_source is not None else "the batch configuration"
    )

    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory '{input_dir}' defined by 'input_dir' in {config_label} does not exist."
        )
    if not input_dir.is_dir():
        raise NotADirectoryError(
            f"The path '{input_dir}' configured via 'input_dir' in {config_label} is not a directory."
        )

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"The path '{output_dir}' configured via 'output_dir' in {config_label} is not a directory."
        )

    try:
        ensure_directory(output_dir)
    except OSError as exc:
        raise OSError(
            f"Could not create output directory '{output_dir}' defined by 'output_dir' in {config_label}: {exc}"
        ) from exc

    summary_path = (
        Path(batch_config.summary_excel)
        if batch_config.summary_excel is not None
        else output_dir / "summary.xlsx"
    )

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
            batch_config=batch_config,
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
            aggregates = update_summary_aggregates(config.summary_excel_path)
            if aggregates is not None:
                print(
                    "[INFO] Updated aggregate statistics in the summary workbook: "
                    f"{config.summary_excel_path}"
                )

    if failures:
        print("\n[SUMMARY] Processing completed with errors:")
        for failed_path, exc in failures:
            print(f"  - {failed_path}: {exc}")
    else:
        print("\n[SUMMARY] All images processed successfully.")


def main(config_path: Optional[Path] = None) -> None:
    """Load the batch configuration and execute the processing pipeline.

    Parameters
    ----------
    config_path : Path or None, optional
        Location of the TOML configuration file. Defaults to
        ``batch_count_intersects_line_grid.toml`` next to this module when not
        provided.
    """
    configure_plot_style()
    resolved_path = Path(config_path) if config_path is not None else Path(__file__).with_suffix(".toml")
    batch_config = load_batch_run_config(resolved_path)
    process_images(batch_config, config_source=resolved_path)


if __name__ == "__main__":
    main()
