"""Batch segmentation script for multiple 2D images.

The module exposes reusable helpers for loading all readable images from an
input directory, applying a segmentation pipeline (denoise -> sharpen ->
threshold -> morphology -> cleanup), and saving binarized results to an output
directory. Filenames are mirrored with a ``_segmented`` suffix.

Configuration is performed directly in Python via :class:`ManualConfiguration`
or through external configuration files that populate the dataclass. Command
line parsing has deliberately been removed to favour explicit, script-driven
setups.
"""

from __future__ import annotations

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


# Import local modules
if __package__ in {None, ""}:
    project_root = Path(__file__).resolve().parents[1]
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
from imppy3d_functions import import_export as imex
from imppy3d_functions import ski_driver_functions as sdrv


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

    Parameters
    ----------
    denoise : Sequence[object]
        Driver arguments forwarded to :func:`ski_driver_functions.apply_driver_denoise`.
        When the selected method is ``"nl_means"`` the second value is interpreted
        as a multiplier for the estimated noise level to derive the ``h`` parameter.
    sharpen : Sequence[object]
        Arguments for :func:`ski_driver_functions.apply_driver_sharpen`.
    threshold : Sequence[object]
        Arguments for :func:`ski_driver_functions.apply_driver_thresholding`.
    morphology : Sequence[object]
        Arguments for :func:`ski_driver_functions.apply_driver_morph`.
    max_hole_size : int
        Largest hole (in pixels) that should be filled during cleanup.
    min_feature_size : int
        Smallest connected component (in pixels) that should be kept.
    invert_grayscale : bool
        Toggle grayscale inversion before segmentation.
    skeletonize_output : bool
        Apply morphological skeletonisation to the final binary mask.
    """

    denoise: Sequence[object]
    sharpen: Sequence[object]
    threshold: Sequence[object]
    morphology: Sequence[object]
    max_hole_size: int
    min_feature_size: int
    invert_grayscale: bool
    skeletonize_output: bool


@dataclass
class ManualConfiguration:
    """Manual execution defaults previously described in the ``USER INPUTS`` block.

    The dataclass acts as single source of truth for scripted runs. Adjust the
    fields directly in Python or populate them via external configuration files
    (e.g. TOML) before invoking :func:`main`.

    Parameters
    ----------
    input_dir : str
        Directory containing the input images for batch processing.
    output_dir : str
        Directory where segmented images will be written.
    invert_grayscale : bool, default=True
        Invert grayscale values during manual processing.
    log_level : str, default='INFO'
        Logging verbosity used for manual execution.
    denoise_method : str, default='nl_means'
        Identifier of the denoising algorithm to apply.
    h_factor : float, default=0.04
        Multiplier applied to the estimated noise sigma to derive the ``h`` value.
    patch_size : int, default=5
        Side length of the neighbourhood used by the denoiser.
    search_distance : int, default=7
        Search window radius for the non-local means filter.
    sharpen_method : str, default='unsharp_mask'
        Name of the sharpening driver to execute.
    sharpen_radius : int, default=2
        Radius parameter passed to the sharpening filter.
    sharpen_amount : float, default=1.0
        Amount parameter forwarded to the sharpening filter.
    threshold_method : str, default='adaptive_threshold'
        Thresholding approach (``'hysteresis_threshold'`` or ``'adaptive_threshold'``).
    hysteresis_low : float, default=25.5
        Lower bound used when hysteresis thresholding is selected.
    hysteresis_high : float, default=51.0
        Upper bound used when hysteresis thresholding is selected.
    adaptive_block_size : int, default=100
        Window size used for adaptive thresholding (must be odd and >= 3).
    adaptive_offset : float, default=-30.0
        Constant offset used by adaptive thresholding.
    morph_operation : int, default=0
        Morphology operation identifier (0=closing, 1=opening, 2=dilation, 3=erosion).
    morph_footprint : int, default=1
        Shape identifier for the morphology footprint (0=square, 1=disk, 2=diamond).
    morph_radius : int, default=1
        Radius of the morphology footprint in pixels.
    max_hole_size : int, default=9
        Maximum hole size (pixels) filled during cleanup.
    min_feature_size : int, default=30
        Minimum feature size (pixels) retained during cleanup.
    apply_skeletonize : bool, default=False
        Apply morphological skeletonisation to the final binary image when set
        to ``True``.
    """

    input_dir: str = (
        r"C:\Users\maxbe\PycharmProjects\GSAT_native\images\native_images"
    )
    output_dir: str = (
        r"C:\Users\maxbe\PycharmProjects\GSAT_native\images\binarised_images"
    )
    invert_grayscale: bool = True
    apply_skeletonize: bool = False
    log_level: str = "INFO"
    denoise_method: str = "nl_means"
    h_factor: float = 0.04
    patch_size: int = 5
    search_distance: int = 7
    sharpen_method: str = "unsharp_mask"
    sharpen_radius: int = 2
    sharpen_amount: float = 1.0
    threshold_method: str = "adaptive_threshold"
    hysteresis_low: float = 25.5
    hysteresis_high: float = 51.0
    adaptive_block_size: int = 100
    adaptive_offset: float = -30.0
    morph_operation: int = 0
    morph_footprint: int = 1
    morph_radius: int = 1
    max_hole_size: int = 9
    min_feature_size: int = 30
    

MANUAL_CONFIGURATION = ManualConfiguration()


@dataclass(frozen=True)
class ExecutionParameters:
    """Runtime options derived from :class:`ManualConfiguration`.

    Parameters
    ----------
    input_dir : Path
        Directory containing the input images for batch processing.
    output_dir : Path
        Directory where segmented images will be written.
    invert_grayscale : bool
        Toggle grayscale inversion before segmentation.
    max_hole_size : int
        Largest hole (in pixels) that should be filled during cleanup.
    min_feature_size : int
        Smallest connected component (in pixels) that should be kept.
    log_level : str
        Logging verbosity used for manual execution.
    """

    input_dir: Path
    output_dir: Path
    invert_grayscale: bool
    max_hole_size: int
    min_feature_size: int
    log_level: str


DEFAULT_PIPELINE = PipelineParameters(
    denoise=("nl_means", 0.8, 5, 7),
    sharpen=("unsharp_mask", 2, 0.3),
    threshold=("hysteresis_threshold", 128, 200),
    morphology=(0, 1, 1),
    max_hole_size=4,
    min_feature_size=64,
    invert_grayscale=False,
    skeletonize_output=False,
)


def build_manual_configuration(
    config: ManualConfiguration | None = None,
) -> tuple[ExecutionParameters, PipelineParameters]:
    """Create manual execution arguments and pipeline parameters.

    Parameters
    ----------
    config : ManualConfiguration or None, default=None
        Manual defaults to use for the run. When ``None``,
        :data:`MANUAL_CONFIGURATION` is used.

    Returns
    -------
    ExecutionParameters
        Execution options for directory handling and logging.
    PipelineParameters
        Fully populated segmentation pipeline configuration.

    Raises
    ------
    ValueError
        If the adaptive block size is smaller than three or if an unsupported
        threshold method is selected.

    Examples
    --------
    >>> manual = ManualConfiguration(input_dir="~/images", output_dir="~/segmented")
    >>> args, pipeline = build_manual_configuration(manual)
    >>> args.output_dir.name
    'segmented'
    """

    manual = config or MANUAL_CONFIGURATION

    input_dir = Path(manual.input_dir).expanduser()
    output_dir = Path(manual.output_dir).expanduser()

    execution = ExecutionParameters(
        input_dir=input_dir,
        output_dir=output_dir,
        invert_grayscale=bool(manual.invert_grayscale),
        max_hole_size=int(manual.max_hole_size),
        min_feature_size=int(manual.min_feature_size),
        log_level=str(manual.log_level),
    )

    threshold_method = str(manual.threshold_method)
    if threshold_method == "hysteresis_threshold":
        threshold_params = (
            threshold_method,
            int(manual.hysteresis_low),
            int(manual.hysteresis_high),
        )
    elif threshold_method == "adaptive_threshold":
        block_size = int(manual.adaptive_block_size)
        if block_size < 3:
            raise ValueError("adaptive_block_size must be >= 3 for manual execution.")
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
            float(manual.adaptive_offset),
        )
    else:  # pragma: no cover - defensive guard for manual configuration
        raise ValueError(
            "threshold_method must be either 'hysteresis_threshold' or 'adaptive_threshold'."
        )

    pipeline = PipelineParameters(
        denoise=(
            str(manual.denoise_method),
            float(manual.h_factor),
            int(manual.patch_size),
            int(manual.search_distance),
        ),
        sharpen=(
            str(manual.sharpen_method),
            int(manual.sharpen_radius),
            float(manual.sharpen_amount),
        ),
        threshold=threshold_params,
        morphology=(
            int(manual.morph_operation),
            int(manual.morph_footprint),
            int(manual.morph_radius),
        ),
        max_hole_size=execution.max_hole_size,
        min_feature_size=execution.min_feature_size,
        invert_grayscale=execution.invert_grayscale,
        skeletonize_output=bool(manual.apply_skeletonize),
    )

    return execution, pipeline


def configure_logging(level: str) -> None:
    """Initialise the logging module with a consistent format.

    Parameters
    ----------
    level : str
        Logging level such as ``"INFO"`` or ``"DEBUG"``.

    Examples
    --------
    >>> configure_logging("DEBUG")
    >>> logging.getLogger().level >= logging.DEBUG
    True
    """

    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def collect_image_files(input_dir: Path) -> List[Path]:
    """Return sorted paths to supported images found in ``input_dir``.

    Parameters
    ----------
    input_dir : Path
        Directory that should be scanned for image files.

    Returns
    -------
    list of Path
        Sorted list of file paths matching :data:`ALLOWED_EXTENSIONS`.

    Examples
    --------
    >>> collect_image_files(Path("./nonexistent"))
    []
    """

    return sorted(
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in ALLOWED_EXTENSIONS
    )


def build_pipeline(
    execution: ExecutionParameters,
    base_pipeline: PipelineParameters | None = None,
) -> PipelineParameters:
    """Construct pipeline parameters for the batch segmentation pipeline.

    Parameters
    ----------
    execution : ExecutionParameters
        Execution options describing hole and feature size thresholds as well as
        grayscale inversion.
    base_pipeline : PipelineParameters or None, optional
        Existing pipeline configuration that should be used as template. When
        ``None`` (default) :data:`DEFAULT_PIPELINE` is used.

    Returns
    -------
    PipelineParameters
        Configuration for the segmentation stages.

    Raises
    ------
    ValueError
        If ``max_hole_size`` or ``min_feature_size`` are negative.

    Examples
    --------
    >>> execution = ExecutionParameters(
    ...     input_dir=Path("./in"),
    ...     output_dir=Path("./out"),
    ...     invert_grayscale=True,
    ...     max_hole_size=1,
    ...     min_feature_size=2,
    ...     log_level="INFO",
    ... )
    >>> params = build_pipeline(execution)
    >>> params.max_hole_size
    1
    """

    if execution.max_hole_size < 0:
        raise ValueError("max_hole_size must be non-negative.")
    if execution.min_feature_size < 0:
        raise ValueError("min_feature_size must be non-negative.")

    template = base_pipeline or DEFAULT_PIPELINE

    return PipelineParameters(
        denoise=template.denoise,
        sharpen=template.sharpen,
        threshold=template.threshold,
        morphology=template.morphology,
        max_hole_size=execution.max_hole_size,
        min_feature_size=execution.min_feature_size,
        invert_grayscale=execution.invert_grayscale,
        skeletonize_output=template.skeletonize_output,
    )


def segment_image(image: np.ndarray, params: PipelineParameters) -> np.ndarray:
    """Apply denoising, sharpening, thresholding and cleanup to ``image``.

    Parameters
    ----------
    image : numpy.ndarray
        Two-dimensional grayscale image that should be segmented.
    params : PipelineParameters
        Configuration describing each processing stage.

    Returns
    -------
    numpy.ndarray
        Binarised representation of the segmented image.

    Raises
    ------
    ValueError
        If parameter validation in downstream driver functions fails.

    Examples
    --------
    >>> params = DEFAULT_PIPELINE
    >>> segmented = segment_image(np.zeros((5, 5), dtype=np.uint8), params)
    >>> segmented.shape
    (5, 5)
    """

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

    if params.skeletonize_output:
        logging.debug("Applying skeletonisation to the final mask.")
        working_bool = morph.skeletonize(working_bool)

    return img_as_ubyte(working_bool)


def validate_directory(path: Path) -> None:
    """Ensure that ``path`` exists and refers to a directory.

    Parameters
    ----------
    path : Path
        Directory to validate.

    Raises
    ------
    FileNotFoundError
        If the path does not exist.
    NotADirectoryError
        If the path exists but is not a directory.

    Examples
    --------
    >>> validate_directory(Path("."))
    """

    if not path.exists():
        raise FileNotFoundError(f"Directory does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {path}")


def process_images(
    input_dir: Path,
    output_dir: Path,
    params: PipelineParameters,
) -> int:
    """Segment every supported image inside ``input_dir``.

    Parameters
    ----------
    input_dir : Path
        Directory containing the images to process.
    output_dir : Path
        Destination directory for the segmented images.
    params : PipelineParameters
        Configuration describing how each image should be segmented.

    Returns
    -------
    int
        Number of images that were successfully processed and saved.

    Raises
    ------
    FileNotFoundError
        If ``input_dir`` does not contain any supported images.

    Examples
    --------
    >>> process_images(Path("./images"), Path("./segmented"), DEFAULT_PIPELINE)
    0
    """

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


def main(
    configuration: ManualConfiguration | None = None,
    pipeline: PipelineParameters | None = None,
) -> int:
    """Execute the segmentation pipeline using Python-driven configuration.

    Parameters
    ----------
    configuration : ManualConfiguration or None, optional
        Manual defaults to use for the run. When ``None`` (default)
        :data:`MANUAL_CONFIGURATION` is used.
    pipeline : PipelineParameters or None, optional
        Explicit pipeline configuration. When ``None`` the pipeline is derived
        from ``configuration`` via :func:`build_manual_configuration`.

    Returns
    -------
    int
        Zero on success, non-zero if validation or processing fails.

    Examples
    --------
    >>> custom = ManualConfiguration(input_dir="./images", output_dir="./out")
    >>> main(custom)  # doctest: +SKIP
    0
    """

    manual = configuration or MANUAL_CONFIGURATION
    configure_logging(str(manual.log_level))

    try:
        execution, derived_pipeline = build_manual_configuration(manual)
    except ValueError as exc:
        logging.error("Invalid manual configuration: %s", exc)
        return 1

    params = pipeline or derived_pipeline

    try:
        validate_directory(execution.input_dir)
    except (FileNotFoundError, NotADirectoryError) as exc:
        logging.error("%s", exc)
        return 1

    try:
        processed = process_images(execution.input_dir, execution.output_dir, params)
    except FileNotFoundError as exc:
        logging.error("%s", exc)
        return 1

    if processed == 0:
        logging.error(
            "No readable images were processed from %s. Please verify the input files.",
            execution.input_dir,
        )
        return 1

    logging.info("Successfully processed %d image(s).", processed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
