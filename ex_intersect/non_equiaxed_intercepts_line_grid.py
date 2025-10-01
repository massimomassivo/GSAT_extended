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
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

try:  # pragma: no cover - import shim for Python < 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - executed when stdlib module is missing
    import tomli as tomllib  # type: ignore[assignment]

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
    file_globs: Tuple[str, ...]
    pipeline_overrides: Dict[str, object]
    save_options: pipeline.SaveOptions


def _load_toml(path: Path) -> MutableMapping[str, object]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


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

    allowed_top_level = {"paths", "pipeline", "save", "save_options"}
    _ensure_allowed_keys(data, allowed_top_level, str(path))

    if "save" in data and "save_options" in data:
        raise ValueError(
            f"Configuration file '{path}' must not define both '[save]' and '[save_options]' sections."
        )

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

    pipeline_section = data.get("pipeline", {})
    pipeline_mapping = _ensure_mapping(pipeline_section, key="pipeline", context=str(path))
    pipeline_overrides = _parse_line_grid_overrides(pipeline_mapping, f"{path}::pipeline")

    save_section = data.get("save") or data.get("save_options") or {}
    save_mapping = _ensure_mapping(save_section, key="save", context=str(path))
    normalised_save_mapping = _normalise_save_mapping(save_mapping, str(path))
    save_options = _parse_save_options(normalised_save_mapping, f"{path}::save")

    longitudinal = PlaneConfig(label="L", input_dir=longitudinal_dir, output_dir=output_root / "L")
    transverse = PlaneConfig(label="T", input_dir=transverse_dir, output_dir=output_root / "T")

    return NonEquiaxedConfig(
        longitudinal=longitudinal,
        transverse=transverse,
        file_globs=file_globs,
        pipeline_overrides=pipeline_overrides,
        save_options=save_options,
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
) -> pipeline.LineGridConfig:
    overrides = dict(pipeline_overrides)
    overrides["file_in_path"] = Path(image_path)
    overrides["results_base_dir"] = plane_output_dir
    return pipeline.LineGridConfig(**overrides)


def _process_plane(
    plane: PlaneConfig,
    *,
    file_globs: Sequence[str],
    pipeline_overrides: Mapping[str, object],
    save_options: pipeline.SaveOptions,
    config_context: str,
) -> Tuple[int, int]:
    ensure_directory(plane.output_dir)

    images = _gather_image_files(plane.input_dir, file_globs, context=f"{config_context}::{plane.label}")
    print(f"[INFO] ({plane.label}) Found {len(images)} image(s) in '{plane.input_dir}'.")

    processed = 0
    failures: List[Tuple[Path, Exception]] = []

    for index, image_path in enumerate(images, start=1):
        print(f"[INFO] ({plane.label}) Processing {image_path} ({index}/{len(images)})")
        config = _build_config_for_image(image_path, plane.output_dir, pipeline_overrides)
        options = replace(save_options)
        try:
            pipeline.process_image(config, options)
        except Exception as exc:  # pragma: no cover - runtime diagnostic output
            print(f"[ERROR] ({plane.label}) Failed to process '{image_path}': {exc}")
            traceback.print_exc()
            failures.append((image_path, exc))
            continue
        processed += 1

    if failures:
        print(f"[SUMMARY] ({plane.label}) Completed with {processed} success(es) and {len(failures)} failure(s).")
    else:
        print(f"[SUMMARY] ({plane.label}) Processed all {processed} image(s) successfully.")

    return processed, len(images)


def main(config_path: Optional[Path] = None) -> None:
    """Execute the non-equiaxed processing workflow."""

    configure_plot_style()
    resolved_path = Path(config_path) if config_path is not None else DEFAULT_TOML
    config = load_config(resolved_path)

    print(f"[INFO] Loaded configuration from '{resolved_path}'.")
    print(f"[INFO] Using file patterns: {', '.join(config.file_globs)}")

    total_processed = 0
    total_found = 0

    for plane in (config.longitudinal, config.transverse):
        processed, found = _process_plane(
            plane,
            file_globs=config.file_globs,
            pipeline_overrides=config.pipeline_overrides,
            save_options=config.save_options,
            config_context=str(resolved_path),
        )
        total_processed += processed
        total_found += found

    print(
        "[SUMMARY] Completed non-equiaxed processing: "
        f"processed {total_processed}/{total_found} image(s) across both planes."
    )


if __name__ == "__main__":
    main()
