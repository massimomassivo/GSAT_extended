"""Batch segmentation script for multiple 2D images.

This command line utility loads each readable image from an input directory,
applies a segmentation pipeline (denoise -> sharpen -> threshold ->
morphology -> cleanup), and saves a binarized result to the specified output
directory. Filenames are mirrored with a ``_segmented`` suffix.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

import numpy as np
from skimage import morphology as morph
from skimage import restoration as srest
from skimage.util import img_as_bool, img_as_float, img_as_ubyte
from skimage.util import invert as ski_invert


# Ensure local modules can be imported when the script is executed directly.
REPO_ROOT = Path(__file__).resolve().parents[1]
IMPPY_MODULE_PATH = REPO_ROOT / "imppy3d_functions"
if str(IMPPY_MODULE_PATH) not in sys.path:
    sys.path.insert(0, str(IMPPY_MODULE_PATH))

import import_export as imex  # noqa: E402  (local import after path setup)
import ski_driver_functions as sdrv  # noqa: E402


ALLOWED_EXTENSIONS = {
    ".tif",
    ".tiff",
    ".png",
    ".jpg",
    ".jpeg",
    ".jp2",
    ".bmp",
    ".dib",
    ".pbm",
    ".ppm",
    ".pgm",
    ".pnm",
}


@dataclass(frozen=True)
class PipelineParameters:
    """Container for the segmentation pipeline parameters.

    For the non-local means denoiser, the second value of ``denoise`` is
    interpreted as a multiplier for the estimated noise level (sigma) to
    derive the ``h`` parameter.
    """

    denoise: Sequence[object]
    sharpen: Sequence[object]
    threshold: Sequence[object]
    morphology: Sequence[object]
    max_hole_size: int
    min_feature_size: int
    invert_grayscale: bool


# -------- USER INPUTS --------

# Set this flag to ``True`` to provide all configuration values directly in this
# script (similar to ``batch_segment_single_image.py``). When ``False``, the
# command line interface is used instead.
USE_MANUAL_CONFIGURATION = True

# Provide the directories that should be used for batch processing. The input
# directory must contain the images to be segmented. All supported images will
# be saved to the output directory with "_segmented" appended to the filename.
manual_input_dir = r"C:\Users\maxbe\PycharmProjects\GSAT_native\images\native_images"
manual_output_dir = r"C:\Users\maxbe\PycharmProjects\GSAT_native\images\binarised_images"

# Segmentation should result in the grain boundaries being WHITE. If the
# resultant segmentation illustrates black grain boundaries, then the image
# grayscale values should be inverted after they are imported.
manual_invert_grayscales = True

# Logging verbosity when executing in manual mode. Valid values are the same as
# the command line interface ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
manual_log_level = "INFO"

# -------- NON-LOCAL MEANS DENOISING FILTER --------
# ===== START INPUTS =====
manual_denoise_method = "nl_means"
# ``manual_h_factor`` scales the estimated noise level (sigma) used to
# derive the non-local means ``h`` parameter. A value of 0.8 mirrors the
# default behaviour of the interactive workflow.
manual_h_factor = 0.04
manual_patch_size = 5     # int
manual_search_dist = 7    # int
# ===== END INPUTS =====

# -------- SHARPEN FILTER --------
# ===== START INPUTS =====
manual_sharpen_method = "unsharp_mask"
manual_sharp_radius = 2     # int
manual_sharp_amount = 1.0   # float (aus deinem Lauf)
# ===== END INPUTS =====

# -------- THRESHOLDING --------
# ===== START INPUTS =====
# Supported methods: "hysteresis_threshold", "adaptive_threshold"
manual_threshold_method = "adaptive_threshold"

# Hysteresis thresholding parameters
manual_low_val = 25.5        # int (aus deinem Lauf, wird bei adaptive ignoriert)
manual_high_val = 51.0       # int (aus deinem Lauf, wird bei adaptive ignoriert)

# Adaptive thresholding parameters
manual_adaptive_block_size = 100   # int (aus deinem Lauf)
manual_adaptive_offset = -30.0     # float (aus deinem Lauf)
# ===== END INPUTS =====

# -------- MORPHOLOGICAL OPERATIONS --------
# ===== START INPUTS =====
# 0: binary_closing
# 1: binary_opening
# 2: binary_dilation
# 3: binary_erosion
manual_op_type = 0     # int (binary_closing)

# 0: square
# 1: disk
# 2: diamond
manual_foot_type = 1   # int (disk)

# Kernel radius (pixels)
manual_morph_rad = 1   # int
# ===== END INPUTS =====

# -------- REMOVE PIXEL ISLANDS AND SMALL HOLES --------
# ===== START INPUTS =====
manual_max_hole_sz = 9    # int (aus deinem Lauf)
manual_min_feat_sz = 30   # int (aus deinem Lauf)
# ===== END INPUTS =====


DEFAULT_PIPELINE = PipelineParameters(
    denoise=("nl_means", 0.8, 5, 7),
    sharpen=("unsharp_mask", 2, 0.3),
    threshold=("hysteresis_threshold", 128, 200),
    morphology=(0, 1, 1),
    max_hole_size=4,
    min_feature_size=64,
    invert_grayscale=False,
)


def build_manual_configuration() -> tuple[argparse.Namespace, PipelineParameters]:
    """Create the ``argparse`` namespace and pipeline for manual execution."""

    input_dir = Path(manual_input_dir).expanduser()
    output_dir = Path(manual_output_dir).expanduser()

    args = argparse.Namespace(
        input_dir=input_dir,
        output_dir=output_dir,
        invert=bool(manual_invert_grayscales),
        max_hole_size=int(manual_max_hole_sz),
        min_feature_size=int(manual_min_feat_sz),
        log_level=str(manual_log_level),
    )

    threshold_method = str(manual_threshold_method)
    if threshold_method == "hysteresis_threshold":
        threshold_params = (
            threshold_method,
            int(manual_low_val),
            int(manual_high_val),
        )
    elif threshold_method == "adaptive_threshold":
        block_size = int(manual_adaptive_block_size)
        if block_size < 3:
            raise ValueError("manual_adaptive_block_size must be >= 3.")
        if block_size % 2 == 0:
            logging.debug(
                "Adaptive threshold block size %s is even; incrementing to %s.",
                block_size,
                block_size + 1,
            )
            block_size += 1
        threshold_params = (
            threshold_method,
            block_size,
            float(manual_adaptive_offset),
        )
    else:  # pragma: no cover - defensive guard for manual configuration
        raise ValueError(
            "manual_threshold_method must be either "
            '"hysteresis_threshold" or "adaptive_threshold".'
        )

    pipeline = PipelineParameters(
        denoise=(
            str(manual_denoise_method),
            float(manual_h_factor),
            int(manual_patch_size),
            int(manual_search_dist),
        ),
        sharpen=(
            str(manual_sharpen_method),
            int(manual_sharp_radius),
            float(manual_sharp_amount),
        ),
        threshold=threshold_params,
        morphology=(
            int(manual_op_type),
            int(manual_foot_type),
            int(manual_morph_rad),
        ),
        max_hole_size=args.max_hole_size,
        min_feature_size=args.min_feature_size,
        invert_grayscale=args.invert,
    )

    return args, pipeline


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch segment all readable images in a directory.",
    )
    parser.add_argument(
        "--input-dir",
        required=True,
        type=Path,
        help="Path to the directory containing images to segment.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory where segmented images will be written.",
    )
    parser.add_argument(
        "--invert",
        action="store_true",
        help="Invert grayscale intensities before segmentation.",
    )
    parser.add_argument(
        "--max-hole-size",
        type=int,
        default=DEFAULT_PIPELINE.max_hole_size,
        help="Maximum hole area (pixels) to fill during cleanup.",
    )
    parser.add_argument(
        "--min-feature-size",
        type=int,
        default=DEFAULT_PIPELINE.min_feature_size,
        help="Minimum feature area (pixels) to retain during cleanup.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity level.",
    )
    return parser.parse_args(argv)


def configure_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def collect_image_files(input_dir: Path) -> List[Path]:
    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def build_pipeline(args: argparse.Namespace) -> PipelineParameters:
    if args.max_hole_size < 0:
        raise ValueError("--max-hole-size must be non-negative.")
    if args.min_feature_size < 0:
        raise ValueError("--min-feature-size must be non-negative.")

    return PipelineParameters(
        denoise=DEFAULT_PIPELINE.denoise,
        sharpen=DEFAULT_PIPELINE.sharpen,
        threshold=DEFAULT_PIPELINE.threshold,
        morphology=DEFAULT_PIPELINE.morphology,
        max_hole_size=args.max_hole_size,
        min_feature_size=args.min_feature_size,
        invert_grayscale=args.invert,
    )


def segment_image(image: np.ndarray, params: PipelineParameters) -> np.ndarray:
    """Apply the segmentation pipeline to a single image array."""

    working_img = img_as_ubyte(image)

    if params.invert_grayscale:
        logging.debug("Inverting grayscale intensities.")
        working_img = img_as_ubyte(ski_invert(working_img))

    logging.debug("Applying denoise filter with parameters: %s", params.denoise)
    denoise_params = list(params.denoise)
    if denoise_params and str(denoise_params[0]).lower() == "nl_means":
        h_factor = float(denoise_params[1])
        sigma_est = srest.estimate_sigma(
            img_as_float(working_img), average_sigmas=True, channel_axis=None
        )
        denoise_params[1] = h_factor * sigma_est
        logging.debug(
            "Estimated noise sigma: %.6f; derived h parameter: %.6f",
            sigma_est,
            denoise_params[1],
        )

    working_img = sdrv.apply_driver_denoise(
        working_img, denoise_params, quiet_in=True
    )

    logging.debug("Applying sharpen filter with parameters: %s", params.sharpen)
    working_img = sdrv.apply_driver_sharpen(
        working_img, list(params.sharpen), quiet_in=True
    )

    logging.debug("Applying threshold with parameters: %s", params.threshold)
    working_img = sdrv.apply_driver_thresholding(
        working_img, list(params.threshold), quiet_in=True
    )

    logging.debug("Applying morphology with parameters: %s", params.morphology)
    working_img = sdrv.apply_driver_morph(
        working_img, list(params.morphology), quiet_in=True
    )

    logging.debug(
        "Removing small holes (<= %s px) and features (< %s px).",
        params.max_hole_size,
        params.min_feature_size,
    )
    working_bool = img_as_bool(working_img)
    working_bool = morph.remove_small_holes(
        working_bool, area_threshold=int(params.max_hole_size), connectivity=1
    )
    working_bool = morph.remove_small_objects(
        working_bool, min_size=int(params.min_feature_size), connectivity=1
    )

    return img_as_ubyte(working_bool)


def validate_directory(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")


def process_images(
    input_dir: Path,
    output_dir: Path,
    params: PipelineParameters,
) -> int:
    image_paths = collect_image_files(input_dir)
    if not image_paths:
        raise FileNotFoundError(
            f"No supported image files were found in {input_dir}."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    processed_count = 0
    total = len(image_paths)
    for index, img_path in enumerate(image_paths, start=1):
        logging.info("Processing %s (%d/%d)", img_path.name, index, total)

        img, img_props = imex.load_image(str(img_path), quiet_in=True)
        if img is None:
            logging.error("Skipping %s: unable to read image.", img_path.name)
            continue
        if img.ndim != 2:
            logging.error(
                "Skipping %s: expected a 2D grayscale image but received shape %s.",
                img_path.name,
                img_props[1] if img_props else img.shape,
            )
            continue

        try:
            segmented = segment_image(img, params)
        except Exception as exc:  # pragma: no cover - defensive guard
            logging.exception("Failed to process %s due to error: %s", img_path.name, exc)
            continue

        if segmented.shape != img.shape:
            logging.error(
                "Skipping %s: segmented image shape %s differs from original %s.",
                img_path.name,
                segmented.shape,
                img.shape,
            )
            continue

        output_path = output_dir / f"{img_path.stem}_segmented{img_path.suffix}"
        if not imex.save_image(segmented, str(output_path), quiet_in=True):
            logging.error("Failed to save segmented image for %s.", img_path.name)
            continue

        logging.info("Saved segmented image to %s", output_path)
        processed_count += 1

    return processed_count


def main(argv: Sequence[str] | None = None) -> int:
    if USE_MANUAL_CONFIGURATION:
        configure_logging(str(manual_log_level))
        try:
            args, params = build_manual_configuration()
        except ValueError as exc:
            logging.error("Invalid manual configuration: %s", exc)
            return 1
    else:
        args = parse_args(argv)
        configure_logging(args.log_level)

        try:
            params = build_pipeline(args)
        except ValueError as exc:
            logging.error("%s", exc)
            return 1

    try:
        validate_directory(args.input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logging.error("%s", exc)
        return 1

    try:
        processed = process_images(args.input_dir, args.output_dir, params)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    if processed == 0:
        logging.error(
            "No readable images were processed from %s. Please verify the input files.",
            args.input_dir,
        )
        return 1

    logging.info("Successfully processed %d image(s).", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
