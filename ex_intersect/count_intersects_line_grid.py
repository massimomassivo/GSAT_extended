"""Command-line entry point for the line grid intersection pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Sequence, Tuple

from matplotlib import pyplot as plt

from ex_intersect import line_grid_pipeline as pipeline

MEDIUM_SIZE = 12
BIGGER_SIZE = 14


def configure_plot_style() -> None:
    """Apply a consistent Matplotlib style for generated figures.

    Examples
    --------
    >>> configure_plot_style()
    >>> plt.rcParams['font.size'] == MEDIUM_SIZE
    True
    """

    plt.rc("font", size=MEDIUM_SIZE)
    plt.rc("axes", titlesize=BIGGER_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=MEDIUM_SIZE)
    plt.rc("ytick", labelsize=MEDIUM_SIZE)
    plt.rc("legend", fontsize=MEDIUM_SIZE)
    plt.rc("figure", titlesize=BIGGER_SIZE)


def build_config_from_user_inputs(
    args: Optional[Sequence[str]] = None,
) -> Tuple[pipeline.LineGridConfig, pipeline.SaveOptions]:
    """Create pipeline configuration objects from CLI parameters.

    Parameters
    ----------
    args : Sequence[str] or None, optional
        Custom argument vector to parse. ``None`` falls back to :data:`sys.argv`.

    Returns
    -------
    tuple of (:class:`LineGridConfig`, :class:`SaveOptions`)
        Configuration controlling the measurement pipeline and persistence options.

    Raises
    ------
    SystemExit
        Raised by :func:`argparse.ArgumentParser.parse_args` when invalid arguments
        are supplied.

    Examples
    --------
    >>> cfg, opts = build_config_from_user_inputs(["image.png", "--row-step", "10"])  # doctest: +SKIP
    >>> cfg.row_step
    10
    """

    parser = argparse.ArgumentParser(
        description=(
            "Count grain intercept lengths on a segmented micrograph using "
            "the line-grid intersection pipeline."
        )
    )
    parser.add_argument(
        "input_image",
        nargs="?",
        default="path/to/segmented_image.jpg",
        help="Path to the segmented binary image that should be analysed.",
    )
    parser.add_argument(
        "--results-dir",
        default="path/to/results_directory",
        help=(
            "Directory where result artefacts will be stored. A sub-directory "
            "matching the image stem will be created automatically."
        ),
    )
    parser.add_argument(
        "--summary-excel",
        default=None,
        help=(
            "Optional path to an Excel workbook that should collect the "
            "summary rows from multiple runs."
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

    parsed = parser.parse_args(args)

    config = pipeline.LineGridConfig(
        file_in_path=Path(parsed.input_image),
        results_base_dir=Path(parsed.results_dir),
        summary_excel_path=Path(parsed.summary_excel)
        if parsed.summary_excel
        else None,
        borders_white=parsed.borders_white,
        row_step=parsed.row_step,
        theta_start=parsed.theta_start,
        theta_end=parsed.theta_end,
        n_theta_steps=parsed.theta_steps,
        inclusive_theta_end=parsed.inclusive_theta_end,
        reskeletonize=parsed.reskeletonize,
        scalebar_pixel=parsed.scalebar_pixel,
        scalebar_micrometer=parsed.scalebar_micrometer,
        crop_rows=tuple(parsed.crop_rows),
        crop_cols=tuple(parsed.crop_cols),
    )
    options = pipeline.SaveOptions(
        save_rotated_images=parsed.save_rotated_images,
        save_boxplot=parsed.save_boxplot,
        save_histograms=parsed.save_histograms,
        save_excel=parsed.save_excel,
        append_summary=parsed.append_summary,
        show_plots=parsed.show_plots,
    )
    return config, options


def main(args: Optional[Sequence[str]] = None) -> None:
    """Parse user inputs, configure Matplotlib, and run the pipeline.

    Parameters
    ----------
    args : Sequence[str] or None, optional
        Optional argument vector forwarded to :func:`build_config_from_user_inputs`.

    Raises
    ------
    SystemExit
        Propagated from :func:`build_config_from_user_inputs` when parsing fails.

    Examples
    --------
    >>> main(["tests/data/segmented.png"])  # doctest: +SKIP
    """

    configure_plot_style()
    config, options = build_config_from_user_inputs(args)
    pipeline.process_image(config, options)


if __name__ == "__main__":
    main()
