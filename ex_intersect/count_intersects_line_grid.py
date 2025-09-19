"""Run the line grid intersection pipeline configured via TOML files.

The script expects a ``count_intersects_line_grid.toml`` file next to the
module. The file mirrors the options used by the interactive CLI in older
versions and now drives the behaviour of :func:`main`.

Examples
--------
.. code-block:: python

    configure_plot_style()
    assert plt.rcParams['font.size'] == MEDIUM_SIZE
"""

from __future__ import annotations

# --- Import shim: erlaubt "Run Current File" UND python -m ---
import sys
from pathlib import Path as _Path

# Projektstamm (Ordner, der 'ex_intersect' enthält) auf sys.path legen
_PROJECT_ROOT = _Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
# --- Ende Import shim ---

from pathlib import Path
from typing import Optional, Tuple

from matplotlib import pyplot as plt

from ex_intersect import line_grid_pipeline as pipeline
from ex_intersect.config_loader import load_single_run_config

MEDIUM_SIZE = 12
BIGGER_SIZE = 14

DEFAULT_TOML = Path(__file__).with_suffix(".toml")


def configure_plot_style() -> None:
    """Apply a consistent Matplotlib style for generated figures."""
    plt.rc("font", size=MEDIUM_SIZE)
    plt.rc("axes", titlesize=BIGGER_SIZE)
    plt.rc("axes", labelsize=MEDIUM_SIZE)
    plt.rc("xtick", labelsize=MEDIUM_SIZE)
    plt.rc("ytick", labelsize=MEDIUM_SIZE)
    plt.rc("legend", fontsize=MEDIUM_SIZE)
    plt.rc("figure", titlesize=BIGGER_SIZE)


def load_config(path: Path = DEFAULT_TOML) -> Tuple[pipeline.LineGridConfig, pipeline.SaveOptions]:
    """Load pipeline configuration from ``path``.

    The TOML file must define at least ``input_image`` and ``results_dir``. See
    ``count_intersects_line_grid.toml`` for the full list of supported keys and
    sections.

    Parameters
    ----------
    path:
        Path to the TOML configuration file. Defaults to :data:`DEFAULT_TOML`.

    Returns
    -------
    tuple of (:class:`LineGridConfig`, :class:`SaveOptions`)
        Configuration objects ready to be passed to the processing pipeline.

    Raises
    ------
    FileNotFoundError
        If the configuration file does not exist.
    ValueError
        If the configuration is missing required keys or contains invalid values.
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(
            f"Configuration file '{config_path}' not found. "
            "Expected at least the keys 'input_image' and 'results_dir'."
        )

    try:
        return load_single_run_config(config_path)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid configuration in '{config_path}'. "
            "Ensure keys like 'input_image', 'results_dir' and the optional "
            "[save_options] section are spelled correctly."
        ) from exc


def main(config_path: Optional[Path] = None) -> None:
    """Execute the pipeline using the TOML-driven configuration.

    Parameters
    ----------
    config_path:
        Optional override for the configuration file. When ``None`` the
        function falls back to :data:`DEFAULT_TOML`.
    """
    configure_plot_style()
    config, options = load_config(config_path or DEFAULT_TOML)
    pipeline.process_image(config, options)


if __name__ == "__main__":
    main()
