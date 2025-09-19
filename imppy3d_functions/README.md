# imppy3d_functions

Dieses Paket bündelt wiederverwendbare Hilfsfunktionen und Wrapper, die von den
Segmentierungs- (`ex_segmentation`) und Intersections-Pipelines
(`ex_intersect`) gemeinsam genutzt werden. Jede Funktion erhält sukzessive
Docstrings sind für alle zentralen Treiberfunktionen verfügbar und
ergänzen die tabellarische Übersicht.

## Modulüberblick

| Modul | Schwerpunkt | Beispiel-Funktionen |
| --- | --- | --- |
| `import_export.py` | Ein-/Ausgabe für 2D/3D-Bilddaten, inkl. TIFF-Stacks. | [`load_image`](./import_export.py#L6), [`save_image`](./import_export.py#L868), [`load_image_seq`](./import_export.py#L387) |
| `ski_driver_functions.py` | High-Level-Wrapper für häufige `scikit-image`-Operationen (Thresholding, Morphologie, Filter). | [`apply_driver_thresholding`](./ski_driver_functions.py#L87), [`apply_driver_morph`](./ski_driver_functions.py#L720), [`interact_driver_denoise`](./ski_driver_functions.py#L318) |
| `grain_size_functions.py` | Werkzeuge zur Geometrie- und Intersections-Analyse. | [`find_intersections`](./grain_size_functions.py#L6), [`measure_line_dist`](./grain_size_functions.py#L196), [`mark_segments_on_image`](./grain_size_functions.py#L311) |
| `volume_image_processing.py` | Utility-Funktionen für volumetrische Daten bzw. Padding. | [`pad_image_boundary`](./volume_image_processing.py#L4) |
| `cv_driver_functions.py` | OpenCV-basierte Treiberfunktionen. | [`apply_driver_blur`](./cv_driver_functions.py#L81), [`apply_driver_thresh`](./cv_driver_functions.py#L758) |
| `cv_processing_wrappers.py` | Hilfsfunktionen rund um Histogramm-Equalisierung und Morphologie. | [`normalize_histogram`](./cv_processing_wrappers.py#L156), [`multi_morph`](./cv_processing_wrappers.py#L237) |
| `plt_wrappers.py` | Hilfsfunktionen für standardisierte Matplotlib-Layouts. | [`create_bw_fig`](./plt_wrappers.py#L4), [`create_2_bw_figs`](./plt_wrappers.py#L42) |
| `cv_interactive_processing.py` / `ski_interactive_processing.py` | Interaktive GUI-Elemente zur Parameterfindung. | [`interact_average_filter`](./cv_interactive_processing.py#L16), [`interact_unsharp_mask`](./ski_interactive_processing.py#L17) |

Weitere Module (`__init__.py`, `ski_processing_wrappers.py`) aggregieren
Funktionen für den einfachen Import in den Beispielskripten.

## Wiederkehrende Rückgabestrukturen

* [`load_image`](./import_export.py#L6) liefert eine Liste aus Numpy-Array und
  Metadaten (`[img, img_prop]`), wobei `img_prop` Anzahl Pixel, Form und Datentyp
  enthält und in den Batch-Skripten direkt weitergereicht wird.
* Die `interact_*`-Funktionen (z. B.
  [`interact_driver_thresholding`](./ski_driver_functions.py#L17) und
  [`interact_average_filter`](./cv_interactive_processing.py#L16)) geben sowohl
  das bearbeitete Bild als auch Parameterlisten zurück, die ohne Anpassung in
  die entsprechenden `apply_*`-Funktionen übergeben werden können.
* [`grain_size_functions.measure_line_dist`](./grain_size_functions.py#L196)
  und [`measure_circular_dist`](./grain_size_functions.py#L241) erzeugen
  Listen von Segmentlängen, die durch
  [`convert_2d_list_to_str`](./grain_size_functions.py#L358) für CSV/Excel
  aufbereitet werden.

## Verwendung in den Pipelines

* Die Segmentierungsskripte rufen vor allem
  [`apply_driver_denoise`](./ski_driver_functions.py#L389),
  [`apply_driver_sharpen`](./ski_driver_functions.py#L242) und
  [`apply_driver_thresholding`](./ski_driver_functions.py#L87) auf, um komplexe
  `scikit-image`-Aufrufe zu kapseln und für automatisierte Batch-Jobs zugänglich
  zu machen.
* Die Intersections-Pipeline nutzt
  [`grain_size_functions.find_intersections`](./grain_size_functions.py#L6) zur
  Ermittlung der Schnittpunkte sowie
  [`mark_segments_on_image`](./grain_size_functions.py#L311) für Debug-Visualisierungen.
* Für 3D-Workflows stehen Helfer wie
  [`apply_driver_morph_3d`](./ski_driver_functions.py#L851) bereit, die in
  zukünftigen Projekten zum Einsatz kommen.

Ausführliche Parameter- und Seiteneffektbeschreibungen befinden sich in den
Docstrings der genannten Funktionen (siehe Links in der Tabelle).

## Sonderfälle und Batch-Hinweise

Die Docstrings der `cv_driver_functions` verweisen auf diesen Abschnitt, der die
häufigsten Stolperfallen bündelt:

* **Invertierte Grauwerte / Binärmasken** – Für Segmentierungen, deren
  Grenzen dunkel statt hell sind, kann `apply_driver_thresh` direkt auf
  invertierte Eingaben angewendet werden. Alternativ lässt sich in den
  Segmentierungskonfigurationen (`ManualConfiguration` bzw.
  `PipelineParameters`, siehe
  [`batch_segment_multiple_images.py`](../ex_segmentation/batch_segment_multiple_images.py))
  das Feld `invert_grayscale` setzen.
* **Batch-Ausführung** – Alle `apply_*`-Funktionen akzeptieren `quiet_in=True`,
  um Protokollausgaben in Stapelläufen zu vermeiden. Dies gilt insbesondere für
  `apply_driver_denoise`, das durch große Suchfenster deutlich längere Laufzeiten
  hat.
* **Interaktive Vorschau vs. Skriptnutzung** – Die `interact_*`-Varianten öffnen
  OpenCV-Fenster und blockieren den Thread, bis Enter oder Esc gedrückt wird.
  Für die Dokumentation der zugehörigen Parameterlisten siehe die jeweiligen
  `apply_*`-Docstrings (z. B. [`apply_driver_morph`](./cv_driver_functions.py#L547)).
