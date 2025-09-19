# ex_intersect

Die Skripte in diesem Ordner bilden die Auswertungs-Pipeline für
Liniengitter-Schnittlängen (Grain Size via Intercepts). Segmentierte Binärbilder
werden eingelesen, mit virtuellen Linealen rotiert beprobt und die resultierenden
Segmentlängen statistisch ausgewertet.

## Pipeline-Überblick

1. **Konfiguration laden** – Die CLI von
   [`count_intersects_line_grid.py`](./count_intersects_line_grid.py) erzeugt mit
   [`build_config_from_user_inputs`](./count_intersects_line_grid.py#L29)
   eine [`LineGridConfig`](./line_grid_pipeline.py#L19) samt
   [`SaveOptions`](./line_grid_pipeline.py#L102).
2. **Bild vorbereiten** – [`prepare_image`](./line_grid_pipeline.py#L207)
   importiert das Segmentierungsresultat über
   [`import_export.load_image`](../imppy3d_functions/import_export.py#L6), schneidet
   den relevanten Bereich zu und legt Ausgabepfade an.
3. **Intersektionen messen** – [`measure_line_intersections`](./line_grid_pipeline.py#L270)
   rotiert das Bild, generiert Linienmasken per
   [`volume_image_processing.pad_image_boundary`](../imppy3d_functions/volume_image_processing.py#L4)
   und ermittelt Schnittpunkte über
   [`grain_size_functions.find_intersections`](../imppy3d_functions/grain_size_functions.py#L6).
4. **Statistik berechnen** – [`aggregate_statistics`](./line_grid_pipeline.py#L337)
   fasst die Ergebnisse in [`AngleStatistics`](./line_grid_pipeline.py#L68)
   und [`StatisticsResult`](./line_grid_pipeline.py#L86) zusammen.
5. **Artefakte speichern** – [`save_outputs`](./line_grid_pipeline.py#L413)
   erstellt Diagramme, Excel-Dateien und optionale Rotationsbilder. Die Pfade
   werden über [`SaveArtifacts`](./line_grid_pipeline.py#L116) verwaltet.

Alle Schritte werden durch [`process_image`](./line_grid_pipeline.py#L475)
koordiniert, das sowohl in der CLI als auch in
[`batch_count_intersects_line_grid.py`](./batch_count_intersects_line_grid.py)
zum Einsatz kommt.

## CLI `count_intersects_line_grid.py`

```
python -m ex_intersect.count_intersects_line_grid <input_image> [OPTIONEN]
```

| Flag | Bedeutung | Standardwert |
| --- | --- | --- |
| `input_image` | Pfad zum binären Segmentierungsbild. | – |
| `--results-dir` | Basisverzeichnis für alle Ergebnisartefakte. | `path/to/results_directory` |
| `--summary-excel` | Gemeinsame Arbeitsmappe für zusammengefasste Läufe. | automatisch `<results_dir>/summary.xlsx` |
| `--row-step` | Abstand der analysierten Rasterzeilen. | `20` |
| `--theta-start` / `--theta-end` | Start- bzw. Endwinkel der Grid-Rotation (Grad). | `0.0` / `180.0` |
| `--theta-steps` | Anzahl der zu untersuchenden Drehwinkel. | `6` |
| `--inclusive-theta-end` | Endwinkel explizit einschließen. | `False` |
| `--scalebar-pixel` / `--scalebar-micrometer` | Pixel- bzw. Realmaßstab der Skala. | `464.0` / `50.0` |
| `--crop-rows START END` | Zeilenbereich des auszuwertenden Ausschnitts. | `(0, 1825)` |
| `--crop-cols START END` | Spaltenbereich des auszuwertenden Ausschnitts. | `(0, 2580)` |
| `--borders-black` | Binärbild vor Verarbeitung invertieren. | Grenzen werden als weiß angenommen |
| `--no-reskeletonize` | Vor Auswertung keine zusätzliche Skelettierung. | Reskeletonizing aktiv |
| `--save-rotated-images` | Zwischenschritte für jede Rotation speichern. | deaktiviert |
| `--no-boxplot` / `--no-histograms` | Grafikausgabe unterdrücken. | aktiviert |
| `--no-excel` | Detaillierte Excel-Ausgabe überspringen. | aktiv |
| `--no-summary` | Kein Append in die Summary-Arbeitsmappe. | aktiv |
| `--show-plots` | Matplotlib-Fenster offen lassen. | geschlossen |

## Anwendungsbeispiel

Ein typisches Szenario ist die Bestimmung der mittleren Korngröße einer
warmgewalzten Nickelbasis-Legierung. Nach der Segmentierung wird das binäre Bild
`my_sample_segmented.tif` mit folgender Befehlszeile analysiert:

```
python -m ex_intersect.count_intersects_line_grid \
    data/my_sample_segmented.tif \
    --results-dir results/nickel_batch \
    --scalebar-pixel 512 --scalebar-micrometer 50 \
    --theta-steps 12 --inclusive-theta-end \
    --crop-rows 100 1700 --crop-cols 150 2400
```

Die Pipeline erzeugt daraufhin eine Excel-Datei mit den
[`AngleStatistics`](./line_grid_pipeline.py#L68), eine Histogramm-Grafik sowie
einen aktualisierten Summary-Eintrag über
[`_append_summary_excel`](./line_grid_pipeline.py#L643). Die zugehörigen
Docstrings werden in den jeweiligen Funktionen gepflegt.

## Konfigurationsvorlagen (Task 5)

Für wiederholbare Läufe stehen zwei kommentierte TOML-Dateien als Vorlage zur
Verfügung:

* [`count_intersects_line_grid.toml`](./count_intersects_line_grid.toml)
  beschreibt einen Einzel-Lauf. Innerhalb der Abschnitte `[paths]`, `[pipeline]`
  und `[save_options]` sind Platzhalter eingetragen, die den CLI-Defaults
  entsprechen. Lege für konkrete Projekte eine Kopie dieser Datei an und trage
  die passenden Pfade sowie Parameter ein.
* [`batch_count_intersects_line_grid.toml`](./batch_count_intersects_line_grid.toml)
  bündelt Batch-Auswertungen. Der Abschnitt `[batch]` definiert Ein- und
  Ausgabeverzeichnisse (sowie optional `summary_excel`), während die Einträge in
  `[pipeline]` und `[save_options]` als globale Standardwerte für alle Bilder
  dienen. Für projektspezifische Sammlungen empfiehlt sich ebenfalls eine
  separate Kopie mit angepassten Pfaden.

## Sonderfälle und weiterführende Docstrings

* **Invertierte Masken** – Nutzen Sie das CLI-Flag `--borders-black`, wenn die
  Segmentierung dunkle Grenzen liefert. Details zur Schwellenwertbehandlung
  liefert [`apply_driver_thresh`](../imppy3d_functions/cv_driver_functions.py#L758).
* **Restartefacts entfernen** – Vor der Intersectionsauswertung können kleine
  Inseln mit [`apply_driver_blob_fill`](../imppy3d_functions/cv_driver_functions.py#L918)
  entfernt werden; der Docstring erläutert Flächen-, Rundheits- und
  Aspektverhältnis-Filter.
* **Batch-Läufe** – Für umfangreiche Serien sollte `quiet_in=True` an den
  OpenCV-Treibern gesetzt werden, um Log-Ausgaben zu begrenzen (siehe
  [`apply_driver_denoise`](../imppy3d_functions/cv_driver_functions.py#L1072)).
