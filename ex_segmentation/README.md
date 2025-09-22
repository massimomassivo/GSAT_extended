# ex_segmentation

Dieses Verzeichnis enthält die Referenz-Skripte, mit denen 2D-Mikrostrukturbilder
für nachgelagerte Analysen segmentiert werden. Die enthaltenen Pipelines bauen
auf den Utility-Modulen in [`imppy3d_functions`](../imppy3d_functions) auf und
führen eine wiederholbare Verarbeitungskette aus Denoising, Schärfung,
Schwellwertbildung, Morphologie und Export aus.

## Typischer Workflow

1. **Bilder importieren** – Rohdaten (TIFF, PNG, JPEG …) werden mit
   [`import_export.load_image`](../imppy3d_functions/import_export.py)
   oder [`collect_image_files`](./batch_segment_multiple_images.py)
   eingelesen.
2. **Pipeline konfigurieren** – Parameter werden über die
   [`PipelineParameters`](./batch_segment_multiple_images.py)
   Struktur festgelegt. Dies geschieht direkt im Skript (z. B. via
   [`MANUAL_CONFIGURATION`](./batch_segment_multiple_images.py)) oder über
   externe Konfigurationsdateien, die eine
   [`ManualConfiguration`](./batch_segment_multiple_images.py) befüllen.
3. **Segmentierung ausführen** –
   [`process_images`](./batch_segment_multiple_images.py) bzw.
   [`segment_image`](./batch_segment_multiple_images.py)
   iterieren über alle Eingabebilder, führen die Filterkette aus und erzeugen
   Binärmasken.
4. **Ergebnisse speichern** – Ausgaben werden über
   [`import_export.save_image`](../imppy3d_functions/import_export.py)
   bzw. per Matplotlib-Exports im interaktiven Workflow gespeichert.

## Wichtige Skripte

| Skript | Zweck | Wichtige Funktionen |
| --- | --- | --- |
| `batch_segment_multiple_images.py` | Stapelverarbeitung kompletter Ordner mit identischen Pipelineparametern. | [`PipelineParameters`](./batch_segment_multiple_images.py), [`build_manual_configuration`](./batch_segment_multiple_images.py), [`process_images`](./batch_segment_multiple_images.py) |
| `batch_segment_single_image.py` | Vorlage für reproduzierbare Einzelbild-Segmentierung mit festen Hyperparametern. | [`sdrv.apply_driver_denoise`](../imppy3d_functions/ski_driver_functions.py), [`sdrv.apply_driver_thresholding`](../imppy3d_functions/ski_driver_functions.py), [`sdrv.apply_driver_morph`](../imppy3d_functions/ski_driver_functions.py) |
| `interactive_processing_single_image.py` | Interaktiver Jupyter-/Matplotlib-Workflow zur Schritt-für-Schritt-Anpassung. | [`interact_adaptive_thresholding`](./interactive_processing_single_image.py), [`interact_del_features`](./interactive_processing_single_image.py), [`interact_skeletonize`](./interactive_processing_single_image.py) |

## Parameterreferenz

Die nachfolgenden Tabellen fassen die häufigsten Eingabeparameter zusammen. Die
Spalte *Quelle* verweist auf die Stelle, an der der Parameter gesetzt oder
verarbeitet wird.

### Denoising

| Parameter | Beschreibung | Standardwert | Quelle |
| --- | --- | --- | --- |
| `denoise[0]` | Algorithmuskennung (`"nl_means"`, `"tv_chambolle"` …) | `"nl_means"` | [`PipelineParameters`](./batch_segment_multiple_images.py) |
| `denoise[1]` / `manual_h_factor` | Multiplikator für die geschätzte Rauschstärke und damit das `h` des Filters. | `0.8` (Default), Beispiel `0.04` im Manual-Mode | [`build_manual_configuration`](./batch_segment_multiple_images.py), [`segment_image`](./batch_segment_multiple_images.py) |
| `denoise[2]` | Patchgröße für NL-Means | `5` | [`segment_image`](./batch_segment_multiple_images.py) |
| `denoise[3]` | Suchfenster für NL-Means | `7` | [`segment_image`](./batch_segment_multiple_images.py) |

### Schärfen & Schwellwert

| Parameter | Beschreibung | Standardwert | Quelle |
| --- | --- | --- | --- |
| `sharpen[0]` | Schärfungsfilter (`"unsharp_mask"`) | `"unsharp_mask"` | [`PipelineParameters`](./batch_segment_multiple_images.py) |
| `sharpen[1]` / `manual_sharp_radius` | Radius des Unsharp-Mask-Kernels | `2` | [`build_pipeline`](./batch_segment_multiple_images.py) |
| `sharpen[2]` / `manual_sharp_amount` | Gewichtung des scharfen Anteils | `0.3` | [`build_pipeline`](./batch_segment_multiple_images.py) |
| `threshold[0]` | Thresholding-Verfahren (`"hysteresis_threshold"`, `"adaptive_threshold"`) | `"hysteresis_threshold"` | [`build_pipeline`](./batch_segment_multiple_images.py) |
| `threshold[1:3]` | Hysteresis-Grenzwerte oder adaptive Parameter (Blockgröße, Offset) | `128/200` bzw. `23/-5` | [`segment_image`](./batch_segment_multiple_images.py) |

### Morphologie & Nachbereitung

| Parameter | Beschreibung | Standardwert | Quelle |
| --- | --- | --- | --- |
| `morphology[0]` | Morphologischer Operator (0=Closing, 1=Opening, …) | `0` | [`segment_image`](./batch_segment_multiple_images.py) |
| `morphology[1]` | Structuring-Element-Typ (0=Square, 1=Disk, 2=Diamond) | `1` | [`segment_image`](./batch_segment_multiple_images.py) |
| `morphology[2]` | Kernelradius in Pixeln | `1` | [`segment_image`](./batch_segment_multiple_images.py) |
| `max_hole_size` | Maximale Größe zu füllender Löcher | `4` | [`PipelineParameters`](./batch_segment_multiple_images.py) |
| `min_feature_size` | Kleinste zu erhaltende Komponenten | `64` | [`PipelineParameters`](./batch_segment_multiple_images.py) |
| `invert_grayscale` | Eingangsbild invertieren, falls Grenzen dunkel sind | `False` | [`PipelineParameters`](./batch_segment_multiple_images.py) |
| `skeletonize_output` / `apply_skeletonize` | Wendet eine abschließende Skelettierung auf die Binärmaske an | `False` | [`segment_image`](./batch_segment_multiple_images.py), [`build_manual_configuration`](./batch_segment_multiple_images.py) |

Weitere Parameter (z. B. Plot-/Logging-Einstellungen) werden in den jeweiligen
Skripten dokumentiert und künftig über Docstrings gepflegt.

## Sonderfälle und weiterführende Docstrings

* **Invertierte Kontraste** – Wenn Korngrenzen dunkel erscheinen, kann in der
  [`ManualConfiguration`](./batch_segment_multiple_images.py) bzw. in den
  [`PipelineParameters`](./batch_segment_multiple_images.py) das Feld
  `invert_grayscale` gesetzt werden. Die Auswirkungen auf die
  Schwellenwertlogik beschreibt der Docstring von
  [`apply_driver_thresh`](../imppy3d_functions/cv_driver_functions.py).
* **Batch-Optimierung** – In Stapelläufen sollten die OpenCV-Treiber mit
  `quiet_in=True` aufgerufen werden, um die Ausgabe schlank zu halten. Siehe
  [`apply_driver_denoise`](../imppy3d_functions/cv_driver_functions.py)
  für Details zu Laufzeit und Parameterwahl.
* **Interaktive Feinanpassung** – Die interaktiven Gegenstücke (z. B.
  [`interact_driver_morph`](../imppy3d_functions/cv_driver_functions.py))
  öffnen Fenster mit Trackbars und geben die Parameterlisten exakt in der
  Reihenfolge zurück, die die entsprechenden `apply_*`-Funktionen erwarten.
