"""Process longitudinal and transverse images with the line-grid pipeline.

Two folders are processed—longitudinal (``L``) and transverse (``T``)—based on
a TOML configuration located next to this module.  The script mirrors the
behaviour of :mod:`count_intersects_line_grid` and
:mod:`batch_count_intersects_line_grid` but intentionally omits a command line
interface; adjust the TOML file to change behaviour.  See
``non_equiaxed_intercepts_line_grid.toml`` for an example configuration.
"""

from __future__ import annotations

# --- Import shim: allows "Run Current File" AND `python -m` without breaking package imports ---
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# --- End import shim ---

import traceback
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - import shim for Python < 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - executed when stdlib module is missing
    import tomli as tomllib  # type: ignore[assignment]

import pandas as pd
import numpy as np

from ex_intersect import line_grid_pipeline as pipeline
from ex_intersect.batch_count_intersects_line_grid import ensure_directory
from ex_intersect.config_loader import (
    _ensure_allowed_keys,
    _parse_line_grid_overrides,
    _parse_save_options,
)
from ex_intersect.count_intersects_line_grid import configure_plot_style

DEFAULT_TOML = Path(__file__).with_suffix(".toml")
DEFAULT_FILE_GLOBS: Tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
DEFAULT_MASTER_EXCEL_NAME = "non_equiaxed_per_image.xlsx"

SAVE_KEY_ALIASES = {
    "write_excel": "save_excel",
}


@dataclass(slots=True)
class PlaneConfig:
    """Describe how one plane (L or T) should be processed."""

    label: str
    input_dir: Path
    output_dir: Path


@dataclass(slots=True)
class NonEquiaxedConfig:
    """Container bundling configuration for both planes."""

    longitudinal: PlaneConfig
    transverse: PlaneConfig
    output_root: Path
    file_globs: Tuple[str, ...]
    pipeline_overrides: Dict[str, object]
    save_options: pipeline.SaveOptions
    sample_id: Optional[str]
    master_excel_name: str
    write_master_excel: bool
    write_master_csv: bool
    build_dir_long_deg: int


@dataclass(slots=True)
class PerImageRecord:
    """Summarise the "All Lines" statistics for one processed image."""

    sample_id: str
    plane: str
    image_name: str
    build_dir_long_deg: int
    n_intercepts: int
    lbar_um: float
    s_um: float
    ci95_half_um: float
    ci95_low_um: float
    ci95_high_um: float
    rel_accuracy_pct: float
    astm_g: float
    results_dir: str
    timestamp: str


@dataclass(slots=True)
class PlaneSummaryRecord:
    """Store pooled statistics for a specific metallographic plane."""

    sample_id: str
    plane: str
    source_plane: str
    n_images: int
    n_intercepts: int
    lbar_um: float
    s_um: float
    ci95_half_um: float
    ci95_low_um: float
    ci95_high_um: float
    rel_accuracy_pct: float
    astm_g: float
    expected_orientation_0deg: bool
    build_dir_long_deg: int


@dataclass(slots=True)
class PlaneProcessingResult:
    """Capture outputs produced while processing one plane."""

    processed: int
    found: int
    per_image_records: List[PerImageRecord]
    pooled_distances: np.ndarray
    pooled_inverse_distances: np.ndarray


FORCED_THETA_VALUES: Tuple[float, ...] = (
    22.5,
    45.0,
    67.5,
    90.0,
    112.5,
    135.0,
    157.5,
    180.0,
)


def _load_toml(path: Path) -> MutableMapping[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _compute_pooled_statistics(
    distances: np.ndarray, inverse_distances: np.ndarray
) -> Dict[str, float]:
    """Derive summary metrics for pooled intercept distances."""

    distances = np.asarray(distances, dtype=float)
    inverse_distances = np.asarray(inverse_distances, dtype=float)

    if distances.size == 0:
        return {
            "segment_count": 0,
            "total_length": 0.0,
            "average_length": float("nan"),
            "std_dev": float("nan"),
            "median_length": float("nan"),
            "ci95_halfwidth_um": float("nan"),
            "ci95_low_um": float("nan"),
            "ci95_high_um": float("nan"),
            "rel_accuracy_pct": float("nan"),
            "astm_g": float("nan"),
        }

    total_length = float(np.sum(distances))
    average_length = float(np.mean(distances))
    std_dev = float(np.std(distances))
    median_length = float(np.median(distances))
    astm_g = pipeline.astm_g_from_lbar_um(average_length)

    ci95_half = float("nan")
    ci95_low = float("nan")
    ci95_high = float("nan")
    rel_accuracy = float("nan")
    valid_distances = distances[np.isfinite(distances)]
    n_valid = int(valid_distances.size)
    if n_valid >= 2 and np.isfinite(average_length):
        sample_std = float(np.std(valid_distances, ddof=1))
        ci95_half = pipeline.T95 * sample_std / np.sqrt(n_valid)
        ci95_low = average_length - ci95_half
        ci95_high = average_length + ci95_half
        if average_length > 0:
            rel_accuracy = 100.0 * ci95_half / average_length

    return {
        "segment_count": int(distances.size),
        "total_length": total_length,
        "average_length": average_length,
        "std_dev": std_dev,
        "median_length": median_length,
        "ci95_halfwidth_um": ci95_half,
        "ci95_low_um": ci95_low,
        "ci95_high_um": ci95_high,
        "rel_accuracy_pct": rel_accuracy,
        "astm_g": astm_g,
    }


def _as_path(value: object, *, key: str, context: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"Expected '{key}' in {context} to be a string path, not {type(value).__name__}")
    result = Path(value)
    if not result:
        raise ValueError(f"The entry '{key}' in {context} must not be empty")
    return result


def _ensure_mapping(section: object, *, key: str, context: str) -> Mapping[str, object]:
    if not isinstance(section, Mapping):
        raise TypeError(f"Section '{key}' in {context} must be a table of key/value pairs")
    return section


def _normalise_save_mapping(raw_mapping: Mapping[str, object], context: str) -> Mapping[str, object]:
    """Translate user-friendly aliases into :class:`SaveOptions` keys."""

    mapping: Dict[str, object] = dict(raw_mapping)
    for alias, target in SAVE_KEY_ALIASES.items():
        if alias in mapping and target not in mapping:
            mapping[target] = mapping[alias]
        mapping.pop(alias, None)

    if "write_plots" in raw_mapping:
        value = raw_mapping["write_plots"]
        mapping.setdefault("save_boxplot", value)
        mapping.setdefault("save_histograms", value)
        mapping.pop("write_plots", None)

    return mapping


def _parse_file_globs(values: Optional[Iterable[object]], *, context: str) -> Tuple[str, ...]:
    if values is None:
        return DEFAULT_FILE_GLOBS

    globs: List[str] = []
    for index, pattern in enumerate(values, start=1):
        if not isinstance(pattern, str):
            raise TypeError(
                f"Entries in 'file_globs' within {context} must be strings; item {index} is {type(pattern).__name__}"
            )
        text = pattern.strip()
        if not text:
            raise ValueError(f"Entries in 'file_globs' within {context} must not be empty")
        globs.append(text)

    return tuple(dict.fromkeys(globs)) or DEFAULT_FILE_GLOBS


def load_config(path: Path) -> NonEquiaxedConfig:
    """Load the non-equiaxed processing configuration from ``path``."""

    path = Path(path)
    data = _load_toml(path)

    allowed_top_level = {"paths", "pipeline", "save", "save_options", "meta", "orientation"}
    _ensure_allowed_keys(data, allowed_top_level, str(path))

    if "save" in data and "save_options" in data:
        raise ValueError(
            f"Configuration file '{path}' must not define both '[save]' and '[save_options]' sections."
        )

    sample_id: Optional[str] = None
    if "meta" in data:
        meta_section = _ensure_mapping(data["meta"], key="meta", context=str(path))
        allowed_meta_keys = {"sample_id"}
        _ensure_allowed_keys(meta_section, allowed_meta_keys, f"{path}::meta")
        raw_sample_id = meta_section.get("sample_id")
        if raw_sample_id is not None:
            if not isinstance(raw_sample_id, str):
                raise TypeError(
                    f"Expected 'sample_id' in {path}::meta to be a string, not {type(raw_sample_id).__name__}"
                )
            stripped = raw_sample_id.strip()
            if stripped:
                sample_id = stripped

    paths_section = _ensure_mapping(data.get("paths", {}), key="paths", context=str(path))
    required_paths = {"longitudinal_dir", "transverse_dir", "output_dir"}
    missing = required_paths - set(paths_section)
    if missing:
        raise KeyError(
            f"Missing required key(s) in [paths] of '{path}': {', '.join(sorted(missing))}"
        )

    longitudinal_dir = _as_path(paths_section["longitudinal_dir"], key="longitudinal_dir", context=f"{path}::paths")
    transverse_dir = _as_path(paths_section["transverse_dir"], key="transverse_dir", context=f"{path}::paths")
    output_root = _as_path(paths_section["output_dir"], key="output_dir", context=f"{path}::paths")

    file_globs = _parse_file_globs(paths_section.get("file_globs"), context=f"{path}::paths")

    orientation_section = data.get("orientation")
    if orientation_section is None:
        raise KeyError(f"Missing required '[orientation]' section in '{path}'.")
    orientation_mapping = _ensure_mapping(orientation_section, key="orientation", context=str(path))
    allowed_orientation_keys = {"build_dir_long_deg"}
    _ensure_allowed_keys(orientation_mapping, allowed_orientation_keys, f"{path}::orientation")
    if "build_dir_long_deg" not in orientation_mapping:
        raise KeyError(
            f"Missing required key 'build_dir_long_deg' in [orientation] section of '{path}'."
        )
    raw_build_dir = orientation_mapping["build_dir_long_deg"]
    if not isinstance(raw_build_dir, (int, float)):
        raise TypeError(
            "The entry 'build_dir_long_deg' in "
            f"{path}::orientation must be a number, not {type(raw_build_dir).__name__}"
        )
    build_dir_value = float(raw_build_dir)
    allowed_orientations = {0.0, 90.0, 180.0}
    if not any(abs(build_dir_value - allowed) < 1e-6 for allowed in allowed_orientations):
        raise ValueError(
            "The entry 'build_dir_long_deg' in "
            f"{path}::orientation must be one of 0, 90, or 180 degrees; received {raw_build_dir!r}."
        )
    build_dir_long_deg = int(round(build_dir_value))

    pipeline_section = data.get("pipeline", {})
    pipeline_mapping = _ensure_mapping(pipeline_section, key="pipeline", context=str(path))
    pipeline_overrides = _parse_line_grid_overrides(pipeline_mapping, f"{path}::pipeline")

    save_section = data.get("save") or data.get("save_options") or {}
    save_mapping = _ensure_mapping(save_section, key="save", context=str(path))

    master_excel_name = DEFAULT_MASTER_EXCEL_NAME
    if "master_excel_name" in save_mapping:
        value = save_mapping["master_excel_name"]
        if not isinstance(value, str):
            raise TypeError(
                f"Expected 'master_excel_name' in {path}::save to be a string, not {type(value).__name__}"
            )
        stripped = value.strip()
        if not stripped:
            raise ValueError(
                f"The entry 'master_excel_name' in {path}::save must not be empty"
            )
        master_excel_name = stripped

    write_master_csv = False
    if "write_csv" in save_mapping:
        value = save_mapping["write_csv"]
        if not isinstance(value, bool):
            raise TypeError(
                f"Expected 'write_csv' in {path}::save to be a boolean, not {type(value).__name__}"
            )
        write_master_csv = value

    write_master_excel: Optional[bool] = None
    for key in ("write_excel", "save_excel"):
        if key in save_mapping:
            value = save_mapping[key]
            if not isinstance(value, bool):
                raise TypeError(
                    f"Expected '{key}' in {path}::save to be a boolean, not {type(value).__name__}"
                )
            write_master_excel = value
            break
    if write_master_excel is None:
        write_master_excel = True

    normalised_save_mapping = _normalise_save_mapping(save_mapping, str(path))
    normalised_save_mapping.pop("master_excel_name", None)
    normalised_save_mapping.pop("write_csv", None)
    save_options = _parse_save_options(normalised_save_mapping, f"{path}::save")

    longitudinal = PlaneConfig(label="L", input_dir=longitudinal_dir, output_dir=output_root / "L")
    transverse = PlaneConfig(label="T", input_dir=transverse_dir, output_dir=output_root / "T")

    return NonEquiaxedConfig(
        longitudinal=longitudinal,
        transverse=transverse,
        output_root=output_root,
        file_globs=file_globs,
        pipeline_overrides=pipeline_overrides,
        save_options=save_options,
        sample_id=sample_id,
        master_excel_name=master_excel_name,
        write_master_excel=write_master_excel,
        write_master_csv=write_master_csv,
        build_dir_long_deg=build_dir_long_deg,
    )


def _gather_image_files(directory: Path, patterns: Sequence[str], *, context: str) -> List[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory '{directory}' referenced in {context} does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(
            f"Input path '{directory}' referenced in {context} is not a directory"
        )

    matched: Dict[Path, None] = {}
    for pattern in patterns:
        for candidate in directory.rglob(pattern):
            if candidate.is_file():
                matched.setdefault(candidate.resolve(), None)

    return sorted(matched)


def _build_config_for_image(
    image_path: Path,
    plane_output_dir: Path,
    pipeline_overrides: Mapping[str, object],
    *,
    build_dir_long_deg: int,
) -> pipeline.LineGridConfig:
    overrides = dict(pipeline_overrides)
    overrides["file_in_path"] = Path(image_path)
    overrides["results_base_dir"] = plane_output_dir
    overrides["theta_start"] = FORCED_THETA_VALUES[0]
    overrides["theta_end"] = FORCED_THETA_VALUES[-1]
    overrides["n_theta_steps"] = len(FORCED_THETA_VALUES)
    overrides["inclusive_theta_end"] = True
    overrides["reskeletonize"] = True
    overrides["build_dir_long_deg"] = build_dir_long_deg

    config = pipeline.LineGridConfig(**overrides)
    theta_values = np.linspace(
        config.theta_start,
        config.theta_end,
        int(np.round(config.n_theta_steps)),
        endpoint=config.inclusive_theta_end,
    )
    if not np.allclose(theta_values, FORCED_THETA_VALUES, atol=1e-6):
        raise AssertionError(
            "Line grid configuration produced unexpected rotation angles: "
            f"{theta_values.tolist()}"
        )

    return config


def _derive_results_dir(config: pipeline.LineGridConfig) -> Path:
    suffix = config.file_in_path.suffix.lower().lstrip(".")
    suffix_part = f"_{suffix}" if suffix else ""
    return config.results_base_dir / f"{config.file_in_path.stem}{suffix_part}_results"


def _process_plane(
    plane: PlaneConfig,
    *,
    file_globs: Sequence[str],
    pipeline_overrides: Mapping[str, object],
    save_options: pipeline.SaveOptions,
    config_context: str,
    build_dir_long_deg: int,
) -> PlaneProcessingResult:
    ensure_directory(plane.output_dir)

    images = _gather_image_files(plane.input_dir, file_globs, context=f"{config_context}::{plane.label}")
    print(f"[INFO] ({plane.label}) Found {len(images)} image(s) in '{plane.input_dir}'.")

    processed = 0
    failures: List[Tuple[Path, Exception]] = []
    records: List[PerImageRecord] = []
    pooled_distances: List[np.ndarray] = []
    pooled_inverse: List[np.ndarray] = []

    for index, image_path in enumerate(images, start=1):
        print(f"[INFO] ({plane.label}) Processing {image_path} ({index}/{len(images)})")
        config = _build_config_for_image(
            image_path,
            plane.output_dir,
            pipeline_overrides,
            build_dir_long_deg=build_dir_long_deg,
        )
        options = replace(save_options)
        try:
            statistics, _artifacts = pipeline.process_image(config, options)
        except Exception as exc:  # pragma: no cover - runtime diagnostic output
            print(f"[ERROR] ({plane.label}) Failed to process '{image_path}': {exc}")
            traceback.print_exc()
            failures.append((image_path, exc))
            continue
        processed += 1

        overall_stats = statistics.overall_statistics
        timestamp = datetime.now().isoformat(timespec="seconds")
        results_dir = _derive_results_dir(config)
        adjusted_build_dir_deg = (
            (build_dir_long_deg + 90) % 180 if plane.label == "T" else build_dir_long_deg
        )
        records.append(
            PerImageRecord(
                sample_id="",  # placeholder, filled later
                plane=plane.label,
                image_name=image_path.name,
                build_dir_long_deg=adjusted_build_dir_deg,
                n_intercepts=int(overall_stats.segment_count),
                lbar_um=float(overall_stats.average_length),
                s_um=float(overall_stats.std_dev),
                ci95_half_um=float(overall_stats.ci95_halfwidth_um),
                ci95_low_um=float(overall_stats.ci95_low_um),
                ci95_high_um=float(overall_stats.ci95_high_um),
                rel_accuracy_pct=float(overall_stats.rel_accuracy_pct),
                astm_g=float(overall_stats.astm_g),
                results_dir=str(results_dir),
                timestamp=timestamp,
            )
        )

        pooled_distances.append(
            statistics.distances_df.get("Distances (µm)", pd.Series(dtype=float)).to_numpy(dtype=float).ravel()
        )
        pooled_inverse.append(
            statistics.inverse_distances_df.get("Inverse Distances (1/µm)", pd.Series(dtype=float)).to_numpy(dtype=float).ravel()
        )

    if failures:
        print(f"[SUMMARY] ({plane.label}) Completed with {processed} success(es) and {len(failures)} failure(s).")
    else:
        print(f"[SUMMARY] ({plane.label}) Processed all {processed} image(s) successfully.")

    non_empty_distances = [arr for arr in pooled_distances if arr.size]
    non_empty_inverse = [arr for arr in pooled_inverse if arr.size]
    pooled_distance_array = (
        np.concatenate(non_empty_distances) if non_empty_distances else np.array([], dtype=float)
    )
    pooled_inverse_array = (
        np.concatenate(non_empty_inverse) if non_empty_inverse else np.array([], dtype=float)
    )

    return PlaneProcessingResult(
        processed=processed,
        found=len(images),
        per_image_records=records,
        pooled_distances=pooled_distance_array,
        pooled_inverse_distances=pooled_inverse_array,
    )


def main(config_path: Optional[Path] = None) -> None:
    """Execute the non-equiaxed processing workflow."""

    configure_plot_style()
    resolved_path = Path(config_path) if config_path is not None else DEFAULT_TOML
    config = load_config(resolved_path)

    print(f"[INFO] Loaded configuration from '{resolved_path}'.")
    print(f"[INFO] Using file patterns: {', '.join(config.file_globs)}")
    print(f"[INFO] build_dir_long_deg (L-plane reference): {config.build_dir_long_deg}°")

    ensure_directory(config.output_root)

    total_processed = 0
    total_found = 0
    per_image_records: List[PerImageRecord] = []
    plane_results: Dict[str, PlaneProcessingResult] = {}

    for plane in (config.longitudinal, config.transverse):
        result = _process_plane(
            plane,
            file_globs=config.file_globs,
            pipeline_overrides=config.pipeline_overrides,
            save_options=config.save_options,
            config_context=str(resolved_path),
            build_dir_long_deg=config.build_dir_long_deg,
        )
        total_processed += result.processed
        total_found += result.found
        per_image_records.extend(result.per_image_records)
        plane_results[plane.label] = result

    print(
        "[SUMMARY] Completed non-equiaxed processing: "
        f"processed {total_processed}/{total_found} image(s) across both planes."
    )

    if per_image_records:
        derived_sample_id = config.sample_id or config.output_root.name
        if not derived_sample_id:
            resolved_output = config.output_root.resolve()
            derived_sample_id = resolved_output.name or resolved_output.as_posix()
        for record in per_image_records:
            record.sample_id = derived_sample_id

        plane_summary_records: List[PlaneSummaryRecord] = []
        plane_stats: Dict[str, Tuple[PlaneProcessingResult, Dict[str, float]]] = {}
        for plane_label, result in plane_results.items():
            stats = _compute_pooled_statistics(
                result.pooled_distances, result.pooled_inverse_distances
            )
            plane_stats[plane_label] = (result, stats)

        if "L" in plane_stats:
            l_result, l_stats = plane_stats["L"]
            plane_summary_records.append(
                PlaneSummaryRecord(
                    sample_id=derived_sample_id,
                    plane="l",
                    source_plane="L",
                    n_images=len(l_result.per_image_records),
                    n_intercepts=int(l_stats["segment_count"]),
                    lbar_um=float(l_stats["average_length"]),
                    s_um=float(l_stats["std_dev"]),
                    ci95_half_um=float(l_stats["ci95_halfwidth_um"]),
                    ci95_low_um=float(l_stats["ci95_low_um"]),
                    ci95_high_um=float(l_stats["ci95_high_um"]),
                    rel_accuracy_pct=float(l_stats["rel_accuracy_pct"]),
                    astm_g=float(l_stats["astm_g"]),
                    expected_orientation_0deg=True,
                    build_dir_long_deg=config.build_dir_long_deg,
                )
            )

        if "T" in plane_stats:
            t_result, t_stats = plane_stats["T"]
            plane_summary_records.append(
                PlaneSummaryRecord(
                    sample_id=derived_sample_id,
                    plane="t",
                    source_plane="T",
                    n_images=len(t_result.per_image_records),
                    n_intercepts=int(t_stats["segment_count"]),
                    lbar_um=float(t_stats["average_length"]),
                    s_um=float(t_stats["std_dev"]),
                    ci95_half_um=float(t_stats["ci95_halfwidth_um"]),
                    ci95_low_um=float(t_stats["ci95_low_um"]),
                    ci95_high_um=float(t_stats["ci95_high_um"]),
                    rel_accuracy_pct=float(t_stats["rel_accuracy_pct"]),
                    astm_g=float(t_stats["astm_g"]),
                    expected_orientation_0deg=False,
                    build_dir_long_deg=config.build_dir_long_deg,
                )
            )

        if "L" in plane_stats:
            l_result, l_stats = plane_stats["L"]
            plane_summary_records.append(
                PlaneSummaryRecord(
                    sample_id=derived_sample_id,
                    plane="p",
                    source_plane="L",
                    n_images=len(l_result.per_image_records),
                    n_intercepts=int(l_stats["segment_count"]),
                    lbar_um=float(l_stats["average_length"]),
                    s_um=float(l_stats["std_dev"]),
                    ci95_half_um=float(l_stats["ci95_halfwidth_um"]),
                    ci95_low_um=float(l_stats["ci95_low_um"]),
                    ci95_high_um=float(l_stats["ci95_high_um"]),
                    rel_accuracy_pct=float(l_stats["rel_accuracy_pct"]),
                    astm_g=float(l_stats["astm_g"]),
                    expected_orientation_0deg=True,
                    build_dir_long_deg=config.build_dir_long_deg,
                )
            )

        dataframe = pd.DataFrame(asdict(record) for record in per_image_records)
        ordered_columns = [
            "sample_id",
            "plane",
            "image_name",
            "build_dir_long_deg",
            "n_intercepts",
            "lbar_um",
            "s_um",
            "ci95_half_um",
            "ci95_low_um",
            "ci95_high_um",
            "rel_accuracy_pct",
            "astm_g",
            "results_dir",
            "timestamp",
        ]
        dataframe = dataframe[ordered_columns]

        plane_summary_df = pd.DataFrame(asdict(record) for record in plane_summary_records)
        plane_summary_columns = [
            "sample_id",
            "plane",
            "source_plane",
            "n_images",
            "n_intercepts",
            "lbar_um",
            "s_um",
            "ci95_half_um",
            "ci95_low_um",
            "ci95_high_um",
            "rel_accuracy_pct",
            "astm_g",
            "expected_orientation_0deg",
            "build_dir_long_deg",
        ]
        if not plane_summary_df.empty:
            plane_summary_df = plane_summary_df[plane_summary_columns]

        master_excel_name = config.master_excel_name
        if not master_excel_name.lower().endswith(".xlsx"):
            master_excel_name = f"{master_excel_name}.xlsx"
        excel_path = config.output_root / master_excel_name

        if config.write_master_excel:
            with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
                dataframe.to_excel(writer, index=False, sheet_name="PerImage")
                if not plane_summary_df.empty:
                    plane_summary_df.to_excel(writer, index=False, sheet_name="PlaneSummary")
            print(
                "[SUMMARY] Master Excel table saved to "
                f"'{excel_path}'."
            )

        if config.write_master_csv:
            csv_path = excel_path.with_suffix(".csv")
            dataframe.to_csv(csv_path, index=False)
            print(f"[SUMMARY] Master CSV table saved to '{csv_path}'.")

        l_count = len(plane_results["L"].per_image_records) if "L" in plane_results else 0
        t_count = len(plane_results["T"].per_image_records) if "T" in plane_results else 0
        print(
            "[SUMMARY] Master table contains "
            f"{len(per_image_records)} row(s) (L: {l_count}, T: {t_count})."
        )
    else:
        print("[SUMMARY] No per-image statistics were recorded; master tables were not created.")


if __name__ == "__main__":
    main()
