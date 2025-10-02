from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.widgets import TextBox, Button
from matplotlib import patches as mpatches
from skimage import io


# ---------- Datei-Dialog ----------
def _pick_image_file(initial_dir: Optional[Path] = None) -> Optional[Path]:
    try:
        import tkinter as tk
        from tkinter import filedialog
        root = tk.Tk()
        root.withdraw()
        filetypes = [("Images", "*.png *.jpg *.jpeg *.tif *.tiff *.bmp"), ("All files", "*.*")]
        fname = filedialog.askopenfilename(
            title="Bild für Skalierung/Scalebar auswählen",
            initialdir=str(initial_dir) if initial_dir else None,
            filetypes=filetypes,
        )
        root.destroy()
        return Path(fname) if fname else None
    except Exception:
        return None


# ---------- (1) px/µm mit Ruler messen (Handles oberhalb der Linie) ----------
def interact_measure_scale_ruler(
    image: np.ndarray,
    *,
    init_um: float = 20.0,
    init_y: int = 1940,
    init_x_left: int = 2211,
    init_x_right: int = 2394,
    handle_radius: int = 4,
    handle_offset_px: int = 12,
    window_title: str = "Skalierung messen – px/µm (Ruler)",
) -> Dict[str, Any]:
    """
    Interaktive Messung der Skalierung (px/µm) mit zwei Griffpunkten.

    Bedienung (auch in der GUI eingeblendet)
    ----------------------------------------
    • Maus: Griffe ziehen (nur horizontal, pixelgenau).
    • ←/→ = ±10 px,  Shift+←/→ = ±1 px (Griff horizontal).
    • ↑/↓ = ±10 px,  Shift+↑/↓ = ±1 px (Linie vertikal).
    • Tab: zwischen linkem/rechtem Griff wechseln.
    • TextBox „µm“: Scalebar-Wert in µm eingeben.
    • Enter/Return oder „Übernehmen“: Ergebnis speichern & Fenster schließen.
    • Tipp: **Lupe** (Toolbar) zum Zoomen verwenden.
    """
    img = image
    if img.ndim == 3 and img.shape[2] == 4:
        img = img[..., :3]
    H, W = img.shape[:2]
    is_gray = (img.ndim == 2)

    # Clamp/Parser
    def clamp_x(x: float) -> int: return int(max(0, min(W - 1, round(x))))
    def clamp_y(y_: float) -> int: return int(max(0, min(H - 1, round(y_))))
    def parse_um(text: str) -> float:
        try: return float(text.replace(",", "."))
        except Exception: return 0.0

    # Initiale Lage
    y_line = clamp_y(init_y)
    x1 = clamp_x(init_x_left)
    x2 = clamp_x(init_x_right)

    fig, ax = plt.subplots(figsize=(11, 6.5))
    try:
        fig.canvas.manager.set_window_title(window_title)
        # >>> Deaktiviert Matplotlibs Standard-Keybindings (z. B. ←/→ = Back/Forward)
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    except Exception:
        pass

    plt.subplots_adjust(bottom=0.30)
    ax.set_title("Skalierung messen – px/µm", pad=8)
    ax.set_axis_off()
    ax.imshow(img, cmap="gray" if is_gray else None, interpolation="nearest")

    # Messlinie exakt auf y_line
    (line,) = ax.plot([x1, x2], [y_line, y_line], color="red", linewidth=2, zorder=2)

    # Handles oberhalb der Linie
    y_handle = max(0, y_line - handle_offset_px)
    h_left  = mpatches.Circle((x1, y_handle), handle_radius, facecolor="white",  edgecolor="red", linewidth=1.5, zorder=4)
    h_right = mpatches.Circle((x2, y_handle), handle_radius, facecolor="yellow", edgecolor="red", linewidth=1.5, zorder=4)
    ax.add_patch(h_left); ax.add_patch(h_right)

    # Dünne Verbinder senkrecht zur Linie (nur visuell)
    (conn_L,) = ax.plot([x1, x1], [y_handle, y_line], color="red", linewidth=1, alpha=0.9, zorder=3)
    (conn_R,) = ax.plot([x2, x2], [y_handle, y_line], color="red", linewidth=1, alpha=0.9, zorder=3)

    selected = "right"

    def current_px() -> int:
        return abs(int(round(h_right.center[0]) - round(h_left.center[0])))

    # Status + Hinweise
    status_text = ax.text(
        0.01, 0.01, "",
        transform=ax.transAxes, color="white",
        bbox=dict(facecolor="black", alpha=0.5, edgecolor="none", boxstyle="round,pad=0.3"),
        ha="left", va="bottom", fontsize=10
    )
    hints = (
        "Maus: Griffe ziehen | ←/→=±10, Shift+←/→=±1 | ↑/↓=±10, Shift+↑/↓=±1 | "
        "Tab Griff wechseln | Enter/'Übernehmen' speichern | Lupe (Toolbar) zum Zoomen"
    )
    fig.text(0.5, 0.025, hints, ha="center", va="bottom", fontsize=9)

    # Widgets
    tb_um_ax     = plt.axes([0.12, 0.11, 0.18, 0.08])
    btn_apply_ax = plt.axes([0.32, 0.11, 0.16, 0.08])
    tb_um  = TextBox(tb_um_ax, "µm:", initial=str(init_um))
    btn_ok = Button(btn_apply_ax, "Übernehmen")

    def _update_selection_vis():
        if selected == "left":
            h_left.set_facecolor("yellow"); h_right.set_facecolor("white")
        else:
            h_left.set_facecolor("white");  h_right.set_facecolor("yellow")

    def _sync_visuals():
        nonlocal y_handle
        y_handle = max(0, y_line - handle_offset_px)
        line.set_data([h_left.center[0], h_right.center[0]], [y_line, y_line])
        conn_L.set_data([h_left.center[0], h_left.center[0]], [y_handle, y_line])
        conn_R.set_data([h_right.center[0], h_right.center[0]], [y_handle, y_line])
        h_left.center  = (h_left.center[0],  y_handle)
        h_right.center = (h_right.center[0], y_handle)

    def _update_status():
        px = current_px()
        um_val = parse_um(tb_um.text)
        if um_val > 0:
            px_per_um = px / um_val
            um_per_px = um_val / px if px > 0 else np.inf
            txt = f"px = {px} | µm = {um_val:g} | px/µm = {px_per_um:.4f} | µm/px = {um_per_px:.4f}"
        else:
            txt = f"px = {px} | µm = {um_val:g} | px/µm = — | µm/px = —"
        status_text.set_text(txt)
        fig.canvas.draw_idle()

    _update_selection_vis()
    _sync_visuals()
    _update_status()

    # Maus: nur horizontales Draggen des ausgewählten Handles
    dragging = False

    def _nearest_handle(event) -> Optional[str]:
        if event.inaxes != ax or event.xdata is None or event.ydata is None:
            return None
        ex, ey = event.xdata, event.ydata
        dL = (ex - h_left.center[0])**2 + (ey - h_left.center[1])**2
        dR = (ex - h_right.center[0])**2 + (ey - h_right.center[1])**2
        r2 = (max(6, handle_radius * 2.2))**2
        if dL <= r2 and dL <= dR: return "left"
        if dR <= r2 and dR <  dL: return "right"
        return None

    def _on_press(event):
        nonlocal selected, dragging
        hit = _nearest_handle(event)
        if hit:
            selected = hit
            dragging = True
            _update_selection_vis()
            fig.canvas.draw_idle()

    def _on_release(_):
        nonlocal dragging
        dragging = False

    def _on_motion(event):
        if not dragging or event.inaxes != ax or event.xdata is None:
            return
        ex = clamp_x(event.xdata)  # nur x, pixelgenau
        if selected == "left":
            h_left.center = (ex, h_left.center[1])
        else:
            h_right.center = (ex, h_right.center[1])
        _sync_visuals()
        _update_status()

    fig.canvas.mpl_connect("button_press_event", _on_press)
    fig.canvas.mpl_connect("button_release_event", _on_release)
    fig.canvas.mpl_connect("motion_notify_event", _on_motion)

    # Tastatur: Shift = 1 px, ohne Shift = 10 px
    def _on_key(event):
        nonlocal selected, y_line
        # Matplotlib liefert Keys wie "left", "shift+left", ...
        is_shift = event.key.startswith("shift+")
        step = 1 if is_shift else 10

        if event.key.endswith("tab"):
            selected = "left" if selected == "right" else "right"
            _update_selection_vis()
            return
        if event.key in ("enter", "return"):
            _apply_and_close(None); return

        hx = h_left if selected == "left" else h_right
        x = hx.center[0]

        if event.key.endswith("left"):
            hx.center = (clamp_x(x - step), hx.center[1])
        elif event.key.endswith("right"):
            hx.center = (clamp_x(x + step), hx.center[1])
        elif event.key.endswith("up"):
            y_line = clamp_y(y_line - step)
        elif event.key.endswith("down"):
            y_line = clamp_y(y_line + step)

        _sync_visuals()
        _update_status()

    fig.canvas.mpl_connect("key_press_event", _on_key)

    def _um_submit(_): _update_status()
    tb_um.on_submit(_um_submit)

    def _apply_and_close(_):
        px = current_px()
        um = parse_um(tb_um.text)
        if um <= 0:
            px_per_um = None; um_per_px = None
        else:
            px_per_um = px / um
            um_per_px = (um / px) if px > 0 else np.inf
        print("[RESULT] Skalierung:")
        print(f"         px = {px}")
        print(f"         µm = {um:g}")
        print(f"         px/µm = {px_per_um if px_per_um is not None else '—'}")
        print(f"         µm/px = {um_per_px if um_per_px is not None else '—'}")
        plt.close(fig)

    btn_ok.on_clicked(_apply_and_close)
    fig.canvas.mpl_connect("close_event", lambda e: None)
    plt.show()

    # Rückgabe (letzter Stand)
    px_final = abs(int(round(h_right.center[0]) - round(h_left.center[0])))
    try:
        um_final = float(str(tb_um.text).replace(",", ".")) if tb_um.text else 0.0
    except Exception:
        um_final = 0.0
    if um_final <= 0:
        return {"px": px_final, "um": um_final, "px_per_um": None, "um_per_px": None}
    return {"px": px_final, "um": um_final, "px_per_um": px_final/um_final, "um_per_px": (um_final/px_final) if px_final>0 else np.inf}


# ---------- (2) Balkendicke unten messen ----------
def interact_measure_bottom_bar_pixels(
    image: np.ndarray,
    init_pixels: int = 100,
    *,
    window_title: str = "Scalebar-Dicke messen – nur Pixel zählen",
    show_band: bool = True,
) -> int:
    """
    Interaktiv die Dicke des unteren Balkens bestimmen.

    Tipp: **Lupe** (Toolbar) zum Zoomen verwenden.
    """
    img = image
    if img.ndim == 3 and img.shape[2] == 4: img = img[..., :3]
    H = int(img.shape[0]); is_gray = (img.ndim == 2)

    def _clamp(p: object) -> int:
        try: p = int(round(float(p)))
        except Exception: p = init_pixels
        return max(0, min(int(p), max(0, H - 1)))

    cut_px = _clamp(init_pixels)

    fig, ax = plt.subplots(figsize=(10.8, 6.4))
    try:
        fig.canvas.manager.set_window_title(window_title)
        # Pfeiltasten-Konflikte sicherheitshalber auch hier deaktivieren
        fig.canvas.mpl_disconnect(fig.canvas.manager.key_press_handler_id)
    except Exception:
        pass

    plt.subplots_adjust(bottom=0.27)
    ax.set_title(f"Scalebar-Dicke messen – Vorschau (Höhe={cut_px}px)")
    ax.set_axis_off()
    ax.imshow(img, cmap="gray" if is_gray else None, interpolation="nearest")

    y_line = H - cut_px - 0.5
    line = ax.axhline(y=y_line, color="red", linewidth=2)
    band = ax.axhspan(H - cut_px, H, facecolor="red", alpha=0.2, edgecolor="none") if (show_band and cut_px>0) else None

    tb_ax   = plt.axes([0.12, 0.11, 0.18, 0.08])
    btn_ax  = plt.axes([0.32, 0.11, 0.16, 0.08])
    tb  = TextBox(tb_ax, "Pixel unten:", initial=str(cut_px))
    btn = Button(btn_ax, "Übernehmen")

    fig.text(0.5, 0.02,
             "↑/↓ (±1 px), Shift+↑/↓ (±10 px) – Pixelzahl | Enter/'Übernehmen' speichern | Lupe (Toolbar) zum Zoomen",
             ha="center", va="bottom", fontsize=9)

    def _redraw():
        nonlocal band
        y = H - cut_px - 0.5
        line.set_ydata([y, y])
        if show_band:
            if band is not None: band.remove()
            if cut_px > 0:
                band = ax.axhspan(H - cut_px, H, facecolor="red", alpha=0.2, edgecolor="none")
        ax.set_title(f"Scalebar-Dicke messen – Vorschau (Höhe={cut_px}px)")
        fig.canvas.draw_idle()

    def _update(_=None):
        nonlocal cut_px
        cut_px = _clamp(tb.text); _redraw()
    btn.on_clicked(_update)

    def _on_key(event):
        nonlocal cut_px
        step = 10 if (event.key in ("shift+up","shift+down")) else 1
        if event.key in ("up","shift+up"):
            cut_px = _clamp(cut_px + step); tb.set_val(str(cut_px)); _redraw()
        elif event.key in ("down","shift+down"):
            cut_px = _clamp(cut_px - step); tb.set_val(str(cut_px)); _redraw()
        elif event.key in ("enter","return"):
            _apply(None)
    fig.canvas.mpl_connect("key_press_event", _on_key)

    def _apply(_=None):
        nonlocal cut_px
        cut_px = _clamp(tb.text)
        print(f"[RESULT] Unterer Balken (Scalebar) zu entfernen: {cut_px} px")
        plt.close(fig)
    fig.canvas.mpl_connect("close_event", lambda e: None)

    plt.show()
    return int(cut_px)


# ---------- (3) Gesamt-Workflow mit Datei-Dialog ----------
def measure_scalebar_then_bottom_band(
    image_path: Optional[Path | str] = None,
    *,
    use_file_dialog: bool = True,
    initial_dir: Optional[Path] = None,
    # Standardwerte:
    init_um: float = 20.0,
    init_ruler_y: int = 1940,
    init_ruler_x_left: int = 2211,
    init_ruler_x_right: int = 2394,
    init_pixels: int = 100,
    show_band: bool = True,
) -> Optional[Dict[str, Any]]:
    """
    1) Skalierung (px/µm) messen – dann 2) Balkendicke (px) messen.
    Bei fehlendem Pfad und use_file_dialog=True: Explorer öffnen.
    """
    path: Optional[Path]
    if image_path is None and use_file_dialog:
        path = _pick_image_file(initial_dir=initial_dir)
        if path is None:
            print("[INFO] Keine Datei gewählt – Vorgang abgebrochen.")
            return None
    else:
        path = Path(image_path) if image_path is not None else None
        if path is None:
            print("[ERROR] Kein Bildpfad und Datei-Dialog deaktiviert.")
            return None

    img = io.imread(str(path))

    scale = interact_measure_scale_ruler(
        img,
        init_um=init_um,
        init_y=init_ruler_y,
        init_x_left=init_ruler_x_left,
        init_x_right=init_ruler_x_right,
        handle_radius=4,
        handle_offset_px=12,
    )
    bottom_px = interact_measure_bottom_bar_pixels(img, init_pixels=init_pixels, show_band=show_band)

    print("[SUMMARY]")
    print(f"  Datei: {path}")
    print(f"  Skalierung: px={scale['px']}, µm={scale['um']}, "
          f"px/µm={scale['px_per_um'] if scale['px_per_um'] is not None else '—'}, "
          f"µm/px={scale['um_per_px'] if scale['um_per_px'] is not None else '—'}")
    print(f"  Balkendicke unten: {bottom_px} px")

    return {"scale": scale, "bottom_bar_px": bottom_px}


if __name__ == "__main__":
    # Direktstart → Explorer → Fenster 1 (px/µm) → Fenster 2 (Balkendicke)
    measure_scalebar_then_bottom_band(
        use_file_dialog=True,
        initial_dir=None,
        init_um=20.0,
        init_ruler_y=1940,
        init_ruler_x_left=2211,
        init_ruler_x_right=2394,
        init_pixels=200,
    )
