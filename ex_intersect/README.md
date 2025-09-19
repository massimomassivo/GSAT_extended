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

## Schritt-für-Schritt: Konfiguration über TOML-Dateien

Statt einer umfangreichen CLI befüllen Sie heute zwei kommentierte
Konfigurationsdateien. So lassen sich Projekte reproduzierbar dokumentieren und
im Team teilen.

1. **Vorlage kopieren** – Erstellen Sie je Projekt eine Kopie von
   [`count_intersects_line_grid.toml`](./count_intersects_line_grid.toml) bzw.
   [`batch_count_intersects_line_grid.toml`](./batch_count_intersects_line_grid.toml).
   Bewahren Sie die Originale als Referenz auf.
2. **Eingaben definieren** – Öffnen Sie die Datei für Einzel-Läufe und setzen
   in `[paths]` mindestens `input_image` (Pfad zum segmentierten Binärbild) und
   `results_dir` (Zielordner für Artefakte). Optional weisen Sie
   `summary_excel` einem bestehenden Sammel-Workbook zu.
3. **Messraster einstellen** – Passen Sie im Abschnitt `[pipeline]` Parameter
   wie `row_step`, `theta_start`, `theta_end`, `theta_steps` oder
   `scalebar_pixel`/`scalebar_micrometer` an. Zuschnitte steuern Sie über die
   Paare `crop_rows = [start, ende]` und `crop_cols = [start, ende]`. Setzen Sie
   `borders_white = false`, falls Ihre Segmentierung dunkle Grenzen liefert.
4. **Ausgabeoptionen wählen** – Legen Sie unter `[save_options]` fest, welche
   Artefakte entstehen sollen. Typische Schalter sind `save_rotated_images`,
   `save_boxplot`, `save_histograms`, `save_excel`, `append_summary` und
   `show_plots`.
5. **Batch-Konfiguration pflegen** – Für Serienläufe geben Sie in der Batch-
   Vorlage unter `[batch]` die Verzeichnisse `input_dir`, `output_dir` und
   optional `summary_excel` an. Globale Parameter passen Sie erneut über
   `[pipeline]` an; einzelne Bilder können ihre eigenen Overrides erhalten, wenn
   Sie in der Datei zusätzliche Tabellen wie `[images."sample.tif".pipeline]`
   ergänzen.

Alle Keys entsprechen den Feldern der Klassen
[`LineGridConfig`](./line_grid_pipeline.py#L19) und
[`SaveOptions`](./line_grid_pipeline.py#L102). Ungültige Werte werden beim
Laden mit klaren Fehlermeldungen abgefangen.

## Skripte ausführen und Ergebnisse verstehen

Nachdem die TOML-Dateien angepasst sind, starten Sie die Pipeline direkt über
die Python-Module:

```bash
python -m ex_intersect.count_intersects_line_grid
```

Der Einzel-Lauf liest automatisch `count_intersects_line_grid.toml` aus demselben
Ordner, verarbeitet das in `input_image` referenzierte Bild und legt sämtliche
Artefakte unter `results_dir` ab. Erwartete Ergebnisse sind u. a. Excel-Dateien
mit [`AngleStatistics`](./line_grid_pipeline.py#L68) und
[`StatisticsResult`](./line_grid_pipeline.py#L86), Histogramme bzw. Boxplots
gemäß `save_histograms` und `save_boxplot` sowie – bei aktivierter Option –
Zwischenbilder für jede Rotation.

Starten Sie den Einzel-Lauf aus dem Projektstamm mit
`python -m ex_intersect.count_intersects_line_grid`. Keine zusätzlichen Flags;
alle Parameter werden aus `count_intersects_line_grid.toml` im selben
Verzeichnis wie das Skript gelesen.

Für Serienläufe verwenden Sie analog `python -m
ex_intersect.batch_count_intersects_line_grid`. Das Batch-Skript liest
`batch_count_intersects_line_grid.toml`, iteriert über `batch.input_dir` und
erstellt für jedes Bild eine eigene Ergebnisstruktur unter `batch.output_dir`.
Sammelergebnisse werden – falls konfiguriert – in `batch.summary_excel`
fortgeschrieben. Weitere Argumente sind nicht nötig; sämtliche Einstellungen
stammen aus der TOML-Datei im Skriptverzeichnis.

## Konfigurationsvorlagen (Task 5)

Für wiederholbare Läufe stehen zwei kommentierte TOML-Dateien als Vorlage zur
Verfügung:

* [`count_intersects_line_grid.toml`](./count_intersects_line_grid.toml)
  beschreibt einen Einzel-Lauf. Innerhalb der Abschnitte `[paths]`, `[pipeline]`
  und `[save_options]` sind kommentierte Platzhalter eingetragen, die den
  bisherigen CLI-Defaults entsprechen. Lege für konkrete Projekte eine Kopie
  dieser Datei an und trage die passenden Pfade sowie Parameter ein.
* [`batch_count_intersects_line_grid.toml`](./batch_count_intersects_line_grid.toml)
  bündelt Batch-Auswertungen. Der Abschnitt `[batch]` definiert Ein- und
  Ausgabeverzeichnisse (sowie optional `summary_excel`), während die Einträge in
  `[pipeline]` und `[save_options]` als globale Standardwerte für alle Bilder
  dienen. Für projektspezifische Sammlungen empfiehlt sich ebenfalls eine
  separate Kopie mit angepassten Pfaden.

## Sonderfälle und weiterführende Docstrings

* **Invertierte Masken** – Setzen Sie `borders_white = false`, wenn die
  Segmentierung dunkle Grenzen liefert. Details zur Schwellenwertbehandlung
  liefert [`apply_driver_thresh`](../imppy3d_functions/cv_driver_functions.py#L758).
* **Restartefacts entfernen** – Vor der Intersectionsauswertung können kleine
  Inseln mit [`apply_driver_blob_fill`](../imppy3d_functions/cv_driver_functions.py#L918)
  entfernt werden; der Docstring erläutert Flächen-, Rundheits- und
  Aspektverhältnis-Filter.
* **Batch-Läufe** – Für umfangreiche Serien sollte `quiet_in=True` an den
  OpenCV-Treibern gesetzt werden, um Log-Ausgaben zu begrenzen (siehe
  [`apply_driver_denoise`](../imppy3d_functions/cv_driver_functions.py#L1072)).
