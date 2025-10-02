"""Process longitudinal and transverse images with the line-grid pipeline."""

from __future__ import annotations

# --- Import shim: allows "Run Current File" AND `python -m` without breaking package imports ---
import sys
from pathlib import Path as _Path

_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# --- End import shim ---

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - import shim for Python < 3.11
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - executed when stdlib module is missing
    import tomli as tomllib  # type: ignore[assignment]

import numpy as np
import pandas as pd

from ex_intersect import line_grid_pipeline as pipeline
from ex_intersect.batch_count_intersects_line_grid import ensure_directory
from ex_intersect.config_loader import (
    _parse_line_grid_overrides,
    _parse_save_options,
)
from ex_intersect.count_intersects_line_grid import configure_plot_style

DEFAULT_TOML = Path(__file__).with_suffix(".toml")
DEFAULT_FILE_GLOBS: Tuple[str, ...] = ("*.png", "*.jpg", "*.jpeg", "*.tif", "*.tiff", "*.bmp")
DEFAULT_MASTER_EXCEL_NAME = "non_equiaxed_per_image.xlsx"
SAVE_KEY_ALIASES = {"write_excel": "save_excel"}
PER_IMAGE_COLUMNS = [
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


@dataclass(slots=True)
class PlaneConfig:
    label: str
    input_dir: Path
    output_dir: Path


@dataclass(slots=True)
class NonEquiaxedConfig:
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
    qa: "QAConfig"


@dataclass(slots=True)
class QAConfig:
    enable: bool = False
    random_seed: Optional[int] = None
    dir_name: str = "Quality Assurance Schnittlinien"
    line_width: float = 1.5
    alpha: float = 0.35
    color_L: str = "tab:red"
    color_T: str = "tab:blue"


ROTATION_SET: Tuple[float, ...] = (
    22.5,
    45.0,
    67.5,
    90.0,
    112.5,
    135.0,
    157.5,
    180.0,
)
FORCED_THETA_VALUES: Tuple[float, ...] = ROTATION_SET


def _load_toml(path: Path) -> Dict[str, object]:
    with path.open("rb") as handle:
        return dict(tomllib.load(handle))


def _as_path(value: object, *, section: str, key: str) -> Path:
    if not isinstance(value, str):
        raise TypeError(f"Expected '{key}' in {section} to be a string path")
    text = value.strip()
    if not text:
        raise ValueError(f"Entry '{key}' in {section} must not be empty")
    return Path(text)


def _parse_file_globs(values: Optional[Iterable[object]]) -> Tuple[str, ...]:
    if values is None:
        return DEFAULT_FILE_GLOBS
    globs = []
    for item in values:
        if not isinstance(item, str):
            raise TypeError("Entries in 'file_globs' must be strings")
        text = item.strip()
        if text:
            globs.append(text)
    return tuple(dict.fromkeys(globs)) or DEFAULT_FILE_GLOBS


def _normalise_save_mapping(mapping: Mapping[str, object]) -> Dict[str, object]:
    result = dict(mapping)
    for alias, target in SAVE_KEY_ALIASES.items():
        if alias in result and target not in result:
            result[target] = result[alias]
        result.pop(alias, None)
    if "write_plots" in result:
        value = result["write_plots"]
        result.setdefault("save_boxplot", value)
        result.setdefault("save_histograms", value)
        result.pop("write_plots", None)
    return result


def load_config(path: Path) -> NonEquiaxedConfig:
    path = Path(path)
    data = _load_toml(path)

    paths_section = data.get("paths") or {}
    if not isinstance(paths_section, Mapping):
        raise TypeError(f"Section '[paths]' in '{path}' must be a table")
    try:
        longitudinal_dir = _as_path(paths_section["longitudinal_dir"], section=f"{path}::paths", key="longitudinal_dir")
        transverse_dir = _as_path(paths_section["transverse_dir"], section=f"{path}::paths", key="transverse_dir")
        output_root = _as_path(paths_section["output_dir"], section=f"{path}::paths", key="output_dir")
    except KeyError as exc:
        raise KeyError(f"Missing required key in [paths]: {exc.args[0]}") from exc
    file_globs = _parse_file_globs(paths_section.get("file_globs"))

    orientation_section = data.get("orientation") or {}
    if not isinstance(orientation_section, Mapping):
        raise TypeError(f"Section '[orientation]' in '{path}' must be a table")
    build_dir_value = orientation_section.get("build_dir_long_deg")
    if build_dir_value is None:
        raise KeyError(f"Missing 'build_dir_long_deg' in [orientation] of '{path}'")
    if isinstance(build_dir_value, bool) or not isinstance(build_dir_value, (int, float)):
        raise TypeError("'build_dir_long_deg' must be numeric")
    build_dir_long_deg = int(round(float(build_dir_value)))
    if build_dir_long_deg not in {0, 90, 180}:
        raise ValueError("'build_dir_long_deg' must be one of {0, 90, 180}")

    meta_section = data.get("meta") or {}
    sample_id = None
    if isinstance(meta_section, Mapping):
        raw = meta_section.get("sample_id")
        if raw is not None:
            if not isinstance(raw, str):
                raise TypeError("'sample_id' must be a string if provided")
            sample_id = raw.strip() or None
    else:
        raise TypeError(f"Section '[meta]' in '{path}' must be a table if present")

    pipeline_section = data.get("pipeline") or {}
    if not isinstance(pipeline_section, Mapping):
        raise TypeError(f"Section '[pipeline]' in '{path}' must be a table")
    pipeline_overrides = _parse_line_grid_overrides(pipeline_section, f"{path}::pipeline")

    save_section = data.get("save") or data.get("save_options") or {}
    if not isinstance(save_section, Mapping):
        raise TypeError(f"Section '[save]' in '{path}' must be a table if present")
    master_excel_name = (
        str(save_section.get("master_excel_name", DEFAULT_MASTER_EXCEL_NAME)).strip()
        or DEFAULT_MASTER_EXCEL_NAME
    )
    write_master_csv = bool(save_section.get("write_csv", False))
    write_master_excel = bool(save_section.get("save_excel", save_section.get("write_excel", True)))
    save_mapping = _normalise_save_mapping(
        {
            k: v
            for k, v in save_section.items()
            if k not in {"master_excel_name", "write_csv", "write_excel"}
        }
    )
    save_options = _parse_save_options(save_mapping, f"{path}::save")

    qa_section = data.get("qa") or {}
    if not isinstance(qa_section, Mapping):
        raise TypeError(f"Section '[qa]' in '{path}' must be a table if present")
    qa_config = QAConfig()
    if qa_section:
        enable = bool(qa_section.get("enable", qa_config.enable))
        seed_value = qa_section.get("random_seed", qa_config.random_seed)
        random_seed = None
        if seed_value is not None:
            if isinstance(seed_value, bool) or not isinstance(seed_value, (int, float)):
                raise TypeError("'random_seed' must be an integer")
            random_seed = int(round(float(seed_value)))
        dir_name = str(qa_section.get("dir_name", qa_config.dir_name)).strip() or qa_config.dir_name
        line_width = float(qa_section.get("line_width", qa_config.line_width))
        alpha = float(qa_section.get("alpha", qa_config.alpha))
        color_L = str(qa_section.get("color_L", qa_config.color_L)).strip() or qa_config.color_L
        color_T = str(qa_section.get("color_T", qa_config.color_T)).strip() or qa_config.color_T
        qa_config = QAConfig(enable, random_seed, dir_name, line_width, alpha, color_L, color_T)

    longitudinal = PlaneConfig("L", longitudinal_dir, output_root / "L")
    transverse = PlaneConfig("T", transverse_dir, output_root / "T")

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
        qa=qa_config,
    )


def _gather_image_files(directory: Path, patterns: Sequence[str], *, context: str) -> Tuple[Path, ...]:
    if not directory.exists():
        raise FileNotFoundError(f"Input directory '{directory}' referenced in {context} does not exist")
    if not directory.is_dir():
        raise NotADirectoryError(f"Input path '{directory}' referenced in {context} is not a directory")

    matched: Dict[Path, None] = {}
    for pattern in patterns:
        for candidate in directory.rglob(pattern):
            if candidate.is_file():
                matched.setdefault(candidate.resolve(), None)
    return tuple(sorted(matched))


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
        raise AssertionError(f"Unexpected rotation angles: {theta_values.tolist()}")
    return config


def _derive_results_dir(config: pipeline.LineGridConfig) -> Path:
    suffix = config.file_in_path.suffix.lower().lstrip(".")
    suffix_part = f"_{suffix}" if suffix else ""
    return config.results_base_dir / f"{config.file_in_path.stem}{suffix_part}_results"


def pick_random_theta_for_image(
    image_path: Path, base_seed: Optional[int], *, rng: np.random.Generator
) -> float:
    if base_seed is not None:
        hash_value = hash(image_path.stem) & 0xFFFFFFFF
        seeded_rng = np.random.default_rng(base_seed ^ hash_value)
        return float(seeded_rng.choice(ROTATION_SET))
    return float(rng.choice(ROTATION_SET))


def save_qa_overlay(
    plane: str,
    image_path: Path,
    theta_deg: float,
    row_step_px: int,
    crop_rows: Tuple[int, int],
    crop_cols: Tuple[int, int],
    out_path: Path,
    color: str,
    alpha: float,
    line_width: float,
) -> None:
    import matplotlib.pyplot as plt  # local import to avoid dependency for non-QA runs
    from matplotlib import image as mpimg

    img = mpimg.imread(str(image_path))
    if img.ndim == 3:
        img = img[..., 0]

    row_start, row_end = crop_rows
    col_start, col_end = crop_cols
    img = img[row_start:row_end or None, col_start:col_end or None]
    height, width = img.shape

    theta_rad = np.deg2rad(theta_deg)
    direction = np.array([np.cos(theta_rad), np.sin(theta_rad)], dtype=float)
    normal = np.array([-direction[1], direction[0]], dtype=float)
    centre = np.array([(width - 1) / 2.0, (height - 1) / 2.0], dtype=float)
    corners = np.array(
        [[0.0, 0.0], [width - 1.0, 0.0], [0.0, height - 1.0], [width - 1.0, height - 1.0]],
        dtype=float,
    )
    projections = (corners - centre) @ normal
    c_min = float(np.min(projections))
    c_max = float(np.max(projections))

    row_step_px = max(int(row_step_px), 1)
    start_k = int(np.floor(c_min / row_step_px))
    end_k = int(np.ceil(c_max / row_step_px))

    diagonal = float(np.hypot(width, height))
    figsize = (max(width, 1) / 100.0, max(height, 1) / 100.0)
    fig, ax = plt.subplots(figsize=figsize, dpi=100)
    ax.imshow(img, cmap="gray", interpolation="nearest")
    ax.set_axis_off()

    for k in range(start_k, end_k + 1):
        offset = k * row_step_px
        p0 = centre + offset * normal
        p1 = p0 - diagonal * direction
        p2 = p0 + diagonal * direction
        ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, alpha=alpha, linewidth=line_width)

    ax.text(
        8,
        16,
        f"{plane}  θ={theta_deg:.1f}°",
        color=color,
        fontsize=9,
        ha="left",
        va="center",
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, edgecolor="none"),
    )

    fig.tight_layout(pad=0)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def summarize_lengths_um(lengths: np.ndarray) -> Dict[str, object]:
    a = np.asarray(lengths, float)
    a = a[np.isfinite(a)]
    N = int(a.size)
    if N < 2:
        return dict(N=0, lbar=np.nan, s=np.nan, half=np.nan, low=np.nan, high=np.nan, ra=np.nan, G=np.nan)
    lbar = float(a.mean())
    s = float(a.std(ddof=1))
    half = 1.960 * s / np.sqrt(N)
    low, high = lbar - half, lbar + half
    ra = 100.0 * half / lbar if lbar > 0 else np.nan
    G = pipeline.astm_g_from_lbar_um(lbar)
    return dict(N=N, lbar=lbar, s=s, half=half, low=low, high=high, ra=ra, G=G)


def delta_ci_lrand(L: Dict[str, object], T: Dict[str, object]) -> Dict[str, float]:
    if (
        any(not np.isfinite(v) for v in (L["lbar"], L["s"], T["lbar"], T["s"]))
        or L["N"] < 2
        or T["N"] < 2
    ):
        return dict(lrand=np.nan, half=np.nan, low=np.nan, high=np.nan, ra=np.nan)
    lL, sL, NL = float(L["lbar"]), float(L["s"]), int(L["N"])
    lT, sT, NT = float(T["lbar"]), float(T["s"]), int(T["N"])
    lrand = float((lL * lL * lT) ** (1.0 / 3.0))
    vL = (sL * sL) / (NL * lL * lL)
    vT = (sT * sT) / (NT * lT * lT)
    var_ln = (4.0 * vL + vT) / 9.0
    sigma_ln = float(np.sqrt(var_ln))
    half = 1.960 * sigma_ln * lrand
    return dict(lrand=lrand, half=half, low=lrand - half, high=lrand + half, ra=100.0 * half / lrand)


def _process_plane(
    plane: PlaneConfig,
    *,
    file_globs: Sequence[str],
    pipeline_overrides: Mapping[str, object],
    save_options: pipeline.SaveOptions,
    config_context: str,
    build_dir_long_deg: int,
    qa_config: QAConfig,
    qa_rng: np.random.Generator,
    sample_id: str,
) -> Dict[str, object]:
    ensure_directory(plane.output_dir)
    images = _gather_image_files(plane.input_dir, file_globs, context=f"{config_context}::{plane.label}")
    print(f"[INFO] ({plane.label}) Found {len(images)} image(s)")

    per_image_rows = []
    pooled_lengths = []
    L_180_values: list[float] = []
    L_90_values: list[float] = []
    qa_saved = 0
    processed = 0

    qa_dir: Optional[Path] = None
    if qa_config.enable:
        qa_dir = plane.output_dir / qa_config.dir_name
        ensure_directory(qa_dir)

    for index, image_path in enumerate(images, start=1):
        print(f"[INFO] ({plane.label}) Processing {index}/{len(images)}: {image_path.name}")
        config = _build_config_for_image(
            image_path,
            plane.output_dir,
            pipeline_overrides,
            build_dir_long_deg=build_dir_long_deg,
        )
        try:
            statistics, _ = pipeline.process_image(config, replace(save_options))
        except Exception as exc:  # pragma: no cover - runtime diagnostic
            print(f"[ERROR] ({plane.label}) {image_path.name}: {exc}")
            continue
        processed += 1

        overall = statistics.overall_statistics
        timestamp = datetime.now().isoformat(timespec="seconds")
        results_dir = _derive_results_dir(config)
        adjusted_build_dir = (build_dir_long_deg + 90) % 180 if plane.label == "T" else build_dir_long_deg
        per_image_rows.append(
            dict(
                sample_id=sample_id,
                plane=plane.label,
                image_name=image_path.name,
                build_dir_long_deg=int(adjusted_build_dir),
                n_intercepts=int(overall.segment_count),
                lbar_um=float(overall.average_length),
                s_um=float(overall.std_dev),
                ci95_half_um=float(overall.ci95_halfwidth_um),
                ci95_low_um=float(overall.ci95_low_um),
                ci95_high_um=float(overall.ci95_high_um),
                rel_accuracy_pct=float(overall.rel_accuracy_pct),
                astm_g=float(overall.astm_g),
                results_dir=str(results_dir),
                timestamp=timestamp,
            )
        )

        pooled_lengths.append(
            statistics.distances_df.get("Distances (µm)", pd.Series(dtype=float)).to_numpy(float).ravel()
        )

        if plane.label == "L":
            for angle_stat in statistics.angle_statistics:
                try:
                    ang = float(angle_stat.angle_label)
                except (TypeError, ValueError):
                    continue
                if not np.isfinite(angle_stat.average_length):
                    continue
                value = float(angle_stat.average_length)
                if abs(ang - 180.0) <= 1e-6 or abs(ang) <= 1e-6:
                    L_180_values.append(value)
                elif abs(ang - 90.0) <= 1e-6:
                    L_90_values.append(value)

        if qa_config.enable and qa_dir is not None:
            theta = pick_random_theta_for_image(image_path, qa_config.random_seed, rng=qa_rng)
            color = qa_config.color_L if plane.label == "L" else qa_config.color_T
            output_name = f"{plane.label}_{image_path.stem}_QA_{theta:.1f}deg.png"
            save_qa_overlay(
                plane.label,
                image_path,
                theta,
                int(config.row_step),
                tuple(int(v) for v in config.crop_rows),
                tuple(int(v) for v in config.crop_cols),
                qa_dir / output_name,
                color,
                qa_config.alpha,
                qa_config.line_width,
            )
            qa_saved += 1
            print(f"[QA] ({plane.label}) saved {qa_saved}/{len(images)}")

    print(f"[SUMMARY] ({plane.label}) processed {processed}/{len(images)} image(s)")
    if qa_config.enable:
        print(f"[QA] ({plane.label}) total overlays saved: {qa_saved}")

    combined_lengths = (
        np.concatenate([arr for arr in pooled_lengths if arr.size])
        if any(arr.size for arr in pooled_lengths)
        else np.array([], float)
    )

    return dict(
        processed=processed,
        found=len(images),
        per_image_df=pd.DataFrame(per_image_rows, columns=PER_IMAGE_COLUMNS) if per_image_rows else pd.DataFrame(columns=PER_IMAGE_COLUMNS),
        pooled_lengths_um=combined_lengths,
        L_180_values=tuple(L_180_values) if plane.label == "L" else tuple(),
        L_90_values=tuple(L_90_values) if plane.label == "L" else tuple(),
        qa_overlays_saved=qa_saved,
    )


def _derive_sample_id(config: NonEquiaxedConfig) -> str:
    if config.sample_id:
        return config.sample_id
    candidate = config.output_root.name
    if candidate:
        return candidate
    resolved = config.output_root.resolve()
    return resolved.name or resolved.as_posix()


def main(config_path: Optional[Path] = None) -> None:
    configure_plot_style()
    resolved_path = Path(config_path) if config_path is not None else DEFAULT_TOML
    config = load_config(resolved_path)

    ensure_directory(config.output_root)
    sample_id = _derive_sample_id(config)
    qa_rng = np.random.default_rng(config.qa.random_seed)

    plane_results: Dict[str, Dict[str, object]] = {}
    total_processed = 0
    total_found = 0
    per_image_frames = []

    for plane in (config.longitudinal, config.transverse):
        result = _process_plane(
            plane,
            file_globs=config.file_globs,
            pipeline_overrides=config.pipeline_overrides,
            save_options=config.save_options,
            config_context=str(resolved_path),
            build_dir_long_deg=config.build_dir_long_deg,
            qa_config=config.qa,
            qa_rng=qa_rng,
            sample_id=sample_id,
        )
        plane_results[plane.label] = result
        total_processed += int(result["processed"])
        total_found += int(result["found"])
        if not result["per_image_df"].empty:
            per_image_frames.append(result["per_image_df"])

    print(
        "[SUMMARY] Completed non-equiaxed processing: "
        f"processed {total_processed}/{total_found} image(s) across both planes."
    )

    if not per_image_frames:
        return

    per_image_df = pd.concat(per_image_frames, ignore_index=True)[PER_IMAGE_COLUMNS]

    plane_rows = []
    L_stats = summarize_lengths_um(plane_results.get("L", {}).get("pooled_lengths_um", np.array([], float)))
    T_stats = summarize_lengths_um(plane_results.get("T", {}).get("pooled_lengths_um", np.array([], float)))

    if "L" in plane_results:
        res_L = plane_results["L"]
        row_base = dict(
            sample_id=sample_id,
            source_plane="L",
            n_images=int(res_L["processed"]),
            n_intercepts=int(L_stats["N"]),
            lbar_um=float(L_stats["lbar"]),
            s_um=float(L_stats["s"]),
            ci95_half_um=float(L_stats["half"]),
            ci95_low_um=float(L_stats["low"]),
            ci95_high_um=float(L_stats["high"]),
            rel_accuracy_pct=float(L_stats["ra"]),
            astm_g=float(L_stats["G"]),
            expected_orientation_0deg=True,
            build_dir_long_deg=config.build_dir_long_deg,
        )
        plane_rows.append(dict(row_base, plane="l"))
        plane_rows.append(dict(row_base, plane="p"))

    if "T" in plane_results:
        res_T = plane_results["T"]
        plane_rows.append(
            dict(
                sample_id=sample_id,
                plane="t",
                source_plane="T",
                n_images=int(res_T["processed"]),
                n_intercepts=int(T_stats["N"]),
                lbar_um=float(T_stats["lbar"]),
                s_um=float(T_stats["s"]),
                ci95_half_um=float(T_stats["half"]),
                ci95_low_um=float(T_stats["low"]),
                ci95_high_um=float(T_stats["high"]),
                rel_accuracy_pct=float(T_stats["ra"]),
                astm_g=float(T_stats["G"]),
                expected_orientation_0deg=False,
                build_dir_long_deg=config.build_dir_long_deg,
            )
        )

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
    plane_summary_df = pd.DataFrame(plane_rows, columns=plane_summary_columns)

    res_L = plane_results.get("L")
    res_T = plane_results.get("T")
    sample_row = dict(
        sample_id=sample_id,
        build_dir_long_deg=config.build_dir_long_deg,
        N_L=int(L_stats["N"]),
        N_T=int(T_stats["N"]),
        lbar_L_um=float(L_stats["lbar"]),
        lbar_T_um=float(T_stats["lbar"]),
        s_L_um=float(L_stats["s"]),
        s_T_um=float(T_stats["s"]),
        lbar_rand_um=np.nan,
        ci95_half_um=np.nan,
        ci95_low_um=np.nan,
        ci95_high_um=np.nan,
        rel_accuracy_pct=np.nan,
        ASTM_G_rand=np.nan,
        weights_note="L weighted 2x (l,p)",
        anisotropy_index=np.nan,
    )

    R = delta_ci_lrand(L_stats, T_stats)
    if np.isfinite(R["lrand"]):
        sample_row.update(
            lbar_rand_um=float(R["lrand"]),
            ci95_half_um=float(R["half"]),
            ci95_low_um=float(R["low"]),
            ci95_high_um=float(R["high"]),
            rel_accuracy_pct=float(R["ra"]),
            ASTM_G_rand=float(pipeline.astm_g_from_lbar_um(R["lrand"])),
        )

    if res_L:
        L_180 = res_L["L_180_values"]
        L_90 = res_L["L_90_values"]
        l0 = np.nanmean(L_180) if len(L_180) else np.nan
        l90 = np.nanmean(L_90) if len(L_90) else np.nan
        if np.isfinite(l0) and np.isfinite(l90) and abs(l90) > 1e-12:
            sample_row["anisotropy_index"] = float(l0 / l90)

    sample_summary_columns = [
        "sample_id",
        "build_dir_long_deg",
        "N_L",
        "N_T",
        "lbar_L_um",
        "lbar_T_um",
        "s_L_um",
        "s_T_um",
        "lbar_rand_um",
        "ci95_half_um",
        "ci95_low_um",
        "ci95_high_um",
        "rel_accuracy_pct",
        "ASTM_G_rand",
        "weights_note",
        "anisotropy_index",
    ]
    sample_summary_df = pd.DataFrame([sample_row], columns=sample_summary_columns)

    master_excel_name = config.master_excel_name
    if not master_excel_name.lower().endswith(".xlsx"):
        master_excel_name = f"{master_excel_name}.xlsx"
    excel_path = config.output_root / master_excel_name

    if config.write_master_excel:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            per_image_df.to_excel(writer, index=False, sheet_name="PerImage")
            if not plane_summary_df.empty:
                plane_summary_df.to_excel(writer, index=False, sheet_name="PlaneSummary")
            sample_summary_df.to_excel(writer, index=False, sheet_name="SampleSummary")

    if config.write_master_csv:
        per_image_df.to_csv(excel_path.with_suffix(".csv"), index=False)


if __name__ == "__main__":
    main()
