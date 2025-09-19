# Grain Size Analysis Tools (GSAT)

Accelerate and automate the process of segmenting images of microscopic 
(metallic) grain boundaries, as well as measuring grain sizes using intercept
methods.

Keywords: grain size, python, segmentation, image processing

## Overview

The Grain Size Analysis Tools (GSAT) is a set of Python scripts that provides tools to segment and measure metallic grains, which are traditionally observed in metals with an optical microscope after polishing and etching a metal sample. Additionally, microstructures measured using common scanning electron microscope (SEM) techniques are also suitable for this library, as will be shown below. Historically, measuring grain size has been a manual process where an experimentalist counts grains or grain boundary intersections. In present day, there are automated solutions, many of which require specialized microscropes and/or commercial software. While computational algorithms are mentioned in the literature related to automating the process of measuring grain size, there is scant open-source software available to the public that performs this process. The GSAT aims to fill this niche by providing free and open-source software (FOSS) that offers batch image processing specific to measuring metallic grain sizes. After choosing appropriate parameters, hundreds of images can be automatically processed using batch scripts in just minutes.

The GSAT separates the process of measuring grain size into two steps: 1) the first step is to segment (or binarize) the selected microstructure image in order to isolate the grain boundaries, and 2) the second step is to calculate various statistics about the grain sizes of a segmented microstructure image using one of two types intercept patterns. Examples of scripts that demonstrate how to segment an image, either interactively or as a batch command, can be found in the examples folder, "ex_segmentation". Additionally, the "ex_intersect" folder contains examples of scripts that calculate grain sizes using different intercept patterns, which are described in more detail in [ASTM E112](https://www.astm.org/standards/e112).

## Quick Start

1. **Install dependencies**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. **Run an interactive segmentation example** to tune parameters for your imagery:
   ```bash
   python ex_segmentation/interactive_processing_single_image.py
   ```
3. **Process segmented data with intercept counting** by editing the TOML configuration and running the module entry point:
   1. Kopieren bzw. bearbeiten Sie `ex_intersect/count_intersects_line_grid.toml`
      und setzen Sie `paths.input_image`, `paths.results_dir` sowie weitere
      Parameter unter `[pipeline]` und `[save_options]`.
   2. Starten Sie anschließend die Auswertung:
      ```bash
      python -m ex_intersect.count_intersects_line_grid
      ```
      Für Serienverarbeitung steht `python -m ex_intersect.batch_count_intersects_line_grid`
      bereit; beide Befehle lesen ihre Einstellungen vollständig aus den TOML-Dateien.
   3. Optional: Öffnen Sie `.runme.yaml` in [Runme](https://runme.dev/) oder
      einem kompatiblen Terminal, um die Kommandos ohne Tippen zu starten.

Core helper utilities live in `imppy3d_functions/`; ensure the example scripts can resolve this directory (via the bundled `sys.path.insert` statements) if you relocate files.

## Architecture Overview

GSAT follows a modular workflow that mirrors the physical measurement procedure:

1. **Segmentation** (`ex_segmentation`): raw grayscale micrographs are denoised, thresholded, and morphologically refined to produce binary masks of grain boundaries.
2. **Intersect counting** (`ex_intersect`): intercept patterns are overlaid on the binary masks to enumerate lineal intercepts and record measurements for downstream analysis.
3. **Evaluation and reporting** (`imppy3d_functions` and example notebooks/scripts): aggregated intercept statistics are converted into grain size metrics, saved as CSV files, and visualized for quality control.

The data flows sequentially from segmentation outputs (binary images) into intersection counters (CSV intercept tallies and annotated images) before being summarized into grain size distributions and auxiliary reports. Intermediate artifacts are persisted between stages so that parameter tuning in earlier phases can be repeated without recomputing later analyses.

## Example of Segmentation and Grain Size Measurement

An example of segmenting an image and measuring grain size using the GSAT is shown next; this example, and the scripts used to calculate the results, can also be found in the example folders: "ex_segmentation" and "ex_intersect". The image chosen for this example was taken using backscatter electrons (BSE) via a scanning electron microscope (SEM). The microstructure corresponds Ti-6Al-4V, which is an alpha-beta titanium alloy.

The first step is to segment the image and isolate the grain boundaries. This can be done either interactively or using batch processing. In general, the interactive image processing script will be easier to find good segmentation parameters when processing an image for the first time. Afterwards, if there are more images taken with the same nominal measurement parameters, the same segmentation parameters will usually be acceptable. Therefore, the remaining images can be safely segmented using a batch processing script. 

The provided example scripts in "ex_intersect" will save the overlaid intercept pattern in a new image so that the user can verify that the pattern is acceptable for his or her needs. Furthermore, every intercept (i.e., line segment) between grain boundaries is recorded and saved in a .csv file. The user can then perfrom a more detailed statistical analysis of the grain size distribution, if needed. 

## Installation

The GSAT is a Python library is dependent on existing libraries like Numpy, SciPy, and SciKit-Image. The list of specific dependencies, and how to install them in a Python environment, are described in the ReadMe file found in the "dependencies" folder.

There are also a number of custom Python modules that the GSAT also depends on which are located in the "imppy3d_functions" folder. At the top of each Python script, there is a line of code that adds a system path string to the system that points to the "imppy3d_functions" folder,

  `sys.path.insert(1, '../imppy3d_functions')`

If the provided scripts are moved to a new directory relative to the "imppy3d_functions" folder, be sure to also update this path variable to the new relative location of the "imppy3d_functions" folder.

## Docstring-Konventionen

All new and updated functions should include [NumPy-style docstrings](https://numpydoc.readthedocs.io/en/latest/format.html), organizing information under the standard sections:

- `Parameters`: list every argument with types and concise descriptions.
- `Returns`: describe the returned values and their shapes or units.
- `Raises`: specify the exceptions that may be propagated.
- `Examples` (optional): provide runnable usage snippets when additional clarity is needed.

## README und Docstrings

Verwende das README für den Überblick über den Gesamtworkflow, zentrale Skripte und typische Einstiegspunkte. Detailfragen zu einzelnen Funktionen, Eingabeparametern und Fehlerfällen werden in den zugehörigen Docstrings beantwortet. So ergänzt sich die Dokumentation: Das README führt durch die großen Schritte (Segmentierung → Intersect-Zählung → Auswertung), während die Docstrings die konkrete Implementierung erläutern. Die Ordner-READMEs verlinken direkt auf die wichtigsten Docstrings, u. a. für `apply_driver_thresh`, `apply_driver_blob_fill` und `apply_driver_denoise`.

## Offene Punkte

* Weitere Module in `imppy3d_functions` (z. B. `import_export.py`) sukzessive auf den NumPy-Docstring-Standard umstellen.
* Beispielskripte um kurze Hinweise ergänzen, welche `apply_*`-Treiber bei invertierten Kontrasten den Vorverarbeitungsschritt übernehmen.

## Support
If you encounter any bugs or unintended behavior, please create an "Issue" and report a bug. You can also make a request for new features in this way. 

For questions on how best to use GSAT for a specific application, feel free
to contact Dr. Newell Moser (see below).

## Author

### Lead developer: 
* Dr. Newell Moser, NIST (newell.moser@nist.gov)
