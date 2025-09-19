"""Helpers for loading pipeline configurations from TOML files.

The functions in this module translate TOML configuration files into
``LineGridConfig`` and ``SaveOptions`` instances used by the
``ex_intersect`` package.  They perform strict validation so that common
mistakes (such as typos in option names or invalid data types) are caught
early with human-friendly error messages.

The minimal configuration required for a single run looks like this::

    input_image = "path/to/image.png"
    results_dir = "results"

    [save_options]
    save_histograms = false

You can load the file with :func:`load_single_run_config`:

>>> from pathlib import Path
>>> _ = Path("single_config.toml").write_text(
...     '''input_image = "sample.png"
... results_dir = "./results"
... [save_options]
... save_boxplot = false
... '''
... )
>>> config, save_opts = load_single_run_config(Path("single_config.toml"))
>>> config.file_in_path == Path("sample.png")
True
>>> config.results_base_dir == Path("./results")
True
>>> save_opts.save_boxplot
False
>>> Path("single_config.toml").unlink()

Batch configurations require at least the directories that should be
processed.  Pipeline overrides are specified inside a dedicated
``[pipeline]`` table::

    input_dir = "./images"
    output_dir = "./results"

    [pipeline]
    row_step = 10

>>> _ = Path("batch_config.toml").write_text(
...     '''input_dir = "images"
... output_dir = "batch_results"
... [pipeline]
... theta_steps = 8
... '''
... )
>>> batch_cfg = load_batch_run_config(Path("batch_config.toml"))
>>> batch_cfg.input_dir == Path("images")
True
>>> batch_cfg.line_grid_overrides["n_theta_steps"]
8
>>> Path("batch_config.toml").unlink()

Running this module directly executes the embedded doctests for a quick
sanity check of the parsing logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, MutableMapping, Optional, Tuple

try:  # pragma: no cover - import shim for Python < 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - executed when stdlib module is missing
    import tomli as tomllib  # type: ignore[assignment]

try:  # pragma: no cover - support running doctests as a script
    from .line_grid_pipeline import LineGridConfig, SaveOptions
except ImportError:  # pragma: no cover - fall back to absolute import
    import sys

    _MODULE_DIR = Path(__file__).resolve().parent
    _PROJECT_ROOT = _MODULE_DIR.parent
    if str(_PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(_PROJECT_ROOT))
    imppy_path = _PROJECT_ROOT / "imppy3d_functions"
    if imppy_path.exists() and str(imppy_path) not in sys.path:
        sys.path.insert(0, str(imppy_path))
    from ex_intersect.line_grid_pipeline import LineGridConfig, SaveOptions  # type: ignore

__all__ = [
    "BatchRunConfig",
    "load_single_run_config",
    "load_batch_run_config",
]


@dataclass(slots=True)
class FieldSpec:
    """Describe how a configuration entry should be parsed."""

    key: str
    dest: str
    expected_types: Tuple[type, ...]
    converter: Optional[Callable[[Any], Any]] = None

    def convert(self, value: Any) -> Any:
        """Validate and transform ``value`` according to the specification."""

        if not isinstance(value, self.expected_types):
            expected = " or ".join(t.__name__ for t in self.expected_types)
            raise TypeError(
                f"Value for '{self.key}' must be of type {expected}, "
                f"not {type(value).__name__}"
            )
        if self.converter is None:
            return value
        return self.converter(value)


def _to_float(value: Any) -> float:
    return float(value)


def _to_int_pair(value: Iterable[Any]) -> Tuple[int, int]:
    values = tuple(value)
    if len(values) != 2:
        raise ValueError(
            f"Expected a pair of integers, received {len(values)} values instead"
        )
    try:
        start, end = (int(v) for v in values)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive programming
        raise TypeError("Both entries must be integers") from exc
    return start, end


LINE_GRID_FIELD_SPECS: Tuple[FieldSpec, ...] = (
    FieldSpec("borders_white", "borders_white", (bool,)),
    FieldSpec("row_step", "row_step", (int,)),
    FieldSpec("theta_start", "theta_start", (int, float), _to_float),
    FieldSpec("theta_end", "theta_end", (int, float), _to_float),
    FieldSpec("theta_steps", "n_theta_steps", (int,)),
    FieldSpec("n_theta_steps", "n_theta_steps", (int,)),
    FieldSpec("inclusive_theta_end", "inclusive_theta_end", (bool,)),
    FieldSpec("reskeletonize", "reskeletonize", (bool,)),
    FieldSpec("scalebar_pixel", "scalebar_pixel", (int, float), _to_float),
    FieldSpec(
        "scalebar_micrometer", "scalebar_micrometer", (int, float), _to_float
    ),
    FieldSpec("crop_rows", "crop_rows", (list, tuple), _to_int_pair),
    FieldSpec("crop_cols", "crop_cols", (list, tuple), _to_int_pair),
)

LINE_GRID_OPTION_KEYS = {spec.key for spec in LINE_GRID_FIELD_SPECS}

SAVE_OPTIONS_SPECS: Tuple[FieldSpec, ...] = (
    FieldSpec("save_rotated_images", "save_rotated_images", (bool,)),
    FieldSpec("save_boxplot", "save_boxplot", (bool,)),
    FieldSpec("save_histograms", "save_histograms", (bool,)),
    FieldSpec("save_excel", "save_excel", (bool,)),
    FieldSpec("append_summary", "append_summary", (bool,)),
    FieldSpec("show_plots", "show_plots", (bool,)),
)

SAVE_OPTION_KEYS = {spec.key for spec in SAVE_OPTIONS_SPECS}


def _require_str(data: Mapping[str, Any], key: str, context: str) -> str:
    if key not in data:
        raise KeyError(f"Missing required key '{key}' in {context}")
    value = data[key]
    if not isinstance(value, str):
        raise TypeError(
            f"Expected '{key}' in {context} to be a string, not {type(value).__name__}"
        )
    return value


def _optional_path(data: Mapping[str, Any], key: str, context: str) -> Optional[Path]:
    if key not in data:
        return None
    value = data[key]
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            f"Expected '{key}' in {context} to be a string path or null, "
            f"not {type(value).__name__}"
        )
    return Path(value)


def _ensure_allowed_keys(
    mapping: Mapping[str, Any], allowed_keys: Iterable[str], context: str
) -> None:
    unknown = set(mapping) - set(allowed_keys)
    if unknown:
        keys = ", ".join(sorted(unknown))
        raise KeyError(f"Unknown configuration keys in {context}: {keys}")


def _parse_line_grid_overrides(
    mapping: Mapping[str, Any], context: str
) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    for spec in LINE_GRID_FIELD_SPECS:
        if spec.key not in mapping:
            continue
        if spec.dest in overrides:
            raise ValueError(
                f"Duplicate pipeline option '{spec.dest}' specified in {context}"
            )
        overrides[spec.dest] = spec.convert(mapping[spec.key])
    _ensure_allowed_keys(mapping, LINE_GRID_OPTION_KEYS, context)
    return overrides


def _parse_save_options(mapping: Optional[Mapping[str, Any]], context: str) -> SaveOptions:
    if mapping is None:
        return SaveOptions()
    if not isinstance(mapping, Mapping):
        raise TypeError(f"Section {context} must be a table of key/value pairs")
    _ensure_allowed_keys(mapping, SAVE_OPTION_KEYS, context)
    options: Dict[str, Any] = {}
    for spec in SAVE_OPTIONS_SPECS:
        if spec.key in mapping:
            options[spec.dest] = spec.convert(mapping[spec.key])
    return SaveOptions(**options)


def _load_toml(path: Path) -> MutableMapping[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def load_single_run_config(path: Path) -> Tuple[LineGridConfig, SaveOptions]:
    """Load a single-run configuration from ``path``.

    Parameters
    ----------
    path:
        Path to the TOML file containing the configuration.

    Returns
    -------
    tuple of (:class:`LineGridConfig`, :class:`SaveOptions`)
        Validated configuration objects that can be passed to the pipeline.
    """

    path = Path(path)
    data = _load_toml(path)

    reserved = {"input_image", "results_dir", "summary_excel", "save_options", "pipeline"}
    allowed = reserved | LINE_GRID_OPTION_KEYS
    _ensure_allowed_keys(data, allowed, str(path))

    input_image = Path(_require_str(data, "input_image", str(path)))
    results_dir = Path(_require_str(data, "results_dir", str(path)))
    summary_excel = _optional_path(data, "summary_excel", str(path))

    top_level_overrides = {
        key: value for key, value in data.items() if key in LINE_GRID_OPTION_KEYS
    }
    if "pipeline" in data:
        pipeline_section_obj = data["pipeline"]
        if not isinstance(pipeline_section_obj, Mapping):
            raise TypeError("The 'pipeline' section must contain key/value pairs")
        pipeline_section = dict(top_level_overrides)
        for key, value in pipeline_section_obj.items():
            if key in pipeline_section:
                raise ValueError(
                    f"Option '{key}' defined both at the top level and in the 'pipeline' table"
                )
            pipeline_section[key] = value
    else:
        pipeline_section = top_level_overrides

    overrides = _parse_line_grid_overrides(pipeline_section, f"{path}::pipeline")
    config_kwargs: Dict[str, Any] = {
        "file_in_path": input_image,
        "results_base_dir": results_dir,
        "summary_excel_path": summary_excel,
    }
    config_kwargs.update(overrides)

    save_options_section = data.get("save_options")
    save_options = _parse_save_options(save_options_section, f"{path}::save_options")

    return LineGridConfig(**config_kwargs), save_options


@dataclass(slots=True)
class BatchRunConfig:
    """Configuration bundle used for batch processing."""

    input_dir: Path
    output_dir: Path
    summary_excel: Optional[Path]
    line_grid_overrides: Dict[str, Any]
    save_options: SaveOptions

    def build_line_grid_config(self, image_path: Path) -> LineGridConfig:
        """Create a :class:`LineGridConfig` for ``image_path``."""

        kwargs = dict(self.line_grid_overrides)
        kwargs.setdefault("results_base_dir", self.output_dir)
        if self.summary_excel is not None:
            kwargs.setdefault("summary_excel_path", self.summary_excel)
        kwargs["file_in_path"] = Path(image_path)
        return LineGridConfig(**kwargs)


def load_batch_run_config(path: Path) -> BatchRunConfig:
    """Load a batch-processing configuration from ``path``."""

    path = Path(path)
    data = _load_toml(path)

    reserved = {
        "input_dir",
        "output_dir",
        "summary_excel",
        "save_options",
        "pipeline",
    }
    allowed = reserved | LINE_GRID_OPTION_KEYS
    _ensure_allowed_keys(data, allowed, str(path))

    input_dir = Path(_require_str(data, "input_dir", str(path)))
    output_dir = Path(_require_str(data, "output_dir", str(path)))
    summary_excel = _optional_path(data, "summary_excel", str(path))

    top_level_overrides = {
        key: value for key, value in data.items() if key in LINE_GRID_OPTION_KEYS
    }
    pipeline_section = data.get("pipeline")
    if pipeline_section is None:
        pipeline_mapping: Mapping[str, Any] = top_level_overrides
    else:
        if not isinstance(pipeline_section, Mapping):
            raise TypeError("The 'pipeline' section must contain key/value pairs")
        combined = dict(top_level_overrides)
        for key, value in pipeline_section.items():
            if key in combined:
                raise ValueError(
                    f"Option '{key}' defined both at the top level and in the 'pipeline' table"
                )
            combined[key] = value
        pipeline_mapping = combined

    overrides = _parse_line_grid_overrides(pipeline_mapping, f"{path}::pipeline")

    save_options_section = data.get("save_options")
    save_options = _parse_save_options(save_options_section, f"{path}::save_options")

    return BatchRunConfig(
        input_dir=input_dir,
        output_dir=output_dir,
        summary_excel=summary_excel,
        line_grid_overrides=overrides,
        save_options=save_options,
    )


if __name__ == "__main__":  # pragma: no cover - convenience for manual checks
    import doctest

    doctest.testmod()
