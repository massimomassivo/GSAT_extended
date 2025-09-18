"""Command-line entry point for the line grid intersection pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

from matplotlib import pyplot as plt

from ex_intersect.line_grid_pipeline import (
    AngleStatistics,
    LineGridConfig,
    SaveArtifacts,
    SaveOptions,
    StatisticsResult,
    aggregate_statistics,
    measure_line_intersections,
    prepare_image,
    save_outputs,
)

MEDIUM_SIZE = 12
BIGGER_SIZE = 14


def configure_plot_style() -> None:
    """Apply a consistent Matplotlib style for generated figures."""

    plt.rc("font", size=MEDIUM_SIZE)
    plt.rc("axes", titlesize=BIGGER_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=MEDIUM_SIZE)
    plt.rc("ytick", labelsize=MEDIUM_SIZE)
    plt.rc("legend", fontsize=MEDIUM_SIZE)
    plt.rc("figure", titlesize=BIGGER_SIZE)


def _print_angle_statistics(stat: AngleStatistics) -> None:
    """Print statistics for a single rotation angle."""

    if stat.angle_label == "All Lines":
        header = "\n --- All Lines Grain Size Statistics --- "
    else:
        header = f"\n --- Grain Size Statistics for {stat.angle_label} deg --- "
    print(header)
    print(f"  Total Number of Grain Segments: {stat.segment_count}")
    print(f"  Summed Length of Grain Segments (µm): {stat.total_length:.2f}")
    print(f"  Average Grain Size (µm): {stat.average_length:.2f}")
    print(f"  Median Grain Size (µm): {stat.median_length:.2f}")
    print(f"  Std. Deviation in Grain Size (µm): {stat.std_dev:.2f}")
    print(
        f"  Thickness from Average Inverse Grain Size (µm): {stat.thickness_from_average:.2f}"
    )
    print(
        f"  Thickness from Median Inverse Grain Size (µm): {stat.thickness_from_median:.2f}"
    )


def print_statistics(statistics: StatisticsResult) -> None:
    """Emit a human-readable summary of the computed statistics."""

    print("\n\n ========== RESULTS SUMMARY ========== ")
    for angle_stat in statistics.angle_statistics:
        _print_angle_statistics(angle_stat)
    _print_angle_statistics(statistics.overall_statistics)


def describe_measurements(
    theta_labels: Sequence[str], segment_counts: Sequence[int]
) -> None:
    """Log how many segments were detected per orientation."""

    for label, count in zip(theta_labels, segment_counts):
        print(
            "  Completed processing intersects for the "
            f"{float(label):.2f} deg orientation with {count} segments..."
        )


def run_pipeline(
    config: LineGridConfig, options: Optional[SaveOptions] = None
) -> Tuple[StatisticsResult, SaveArtifacts]:
    """Execute the full measurement pipeline and persist artefacts."""

    print("\nCounting intersect distances for each orientation...")
    prepared = prepare_image(config)
    measurements = measure_line_intersections(prepared, config)
    segment_counts = tuple(len(arr) for arr in measurements.per_theta_distances)
    describe_measurements(measurements.theta_labels, segment_counts)
    statistics = aggregate_statistics(measurements, config)
    print_statistics(statistics)
    artifacts = save_outputs(prepared, measurements, statistics, config, options)
    print("\nScript successfully terminated!")
    return statistics, artifacts


def main() -> None:
    """Example configuration that mirrors the former script defaults."""

    configure_plot_style()
    config = LineGridConfig(
        file_in_path=Path("path/to/segmented_image.jpg"),
        results_base_dir=Path("path/to/results_directory"),
        borders_white=True,
        row_step=20,
        theta_start=0.0,
        theta_end=180.0,
        n_theta_steps=6,
        inclusive_theta_end=False,
        reskeletonize=True,
        scalebar_pixel=464,
        scalebar_micrometer=50,
        crop_rows=(0, 1825),
        crop_cols=(0, 2580),
    )
    options = SaveOptions(show_plots=True)
    run_pipeline(config, options)


if __name__ == "__main__":
    main()

