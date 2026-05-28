#!/usr/bin/env python3
"""Tkinter GUI for exploring Carp Pond simulated bird active space."""

from __future__ import annotations

import argparse
import os
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path



LHFG = 0.2
UHFL = 0.8
SHFG = 1.0

FALLBACK_MIC_LOCS_POINTS = (
    [
        [0.0, 0.0, 6.65 + LHFG],
        [0.0, 0.0, 6.65 + UHFL],
        [0.0, -20.0, 6.75 + LHFG],
        [0.0, -20.0, 6.75 + UHFL],
        [3.0, -48.0, 6.8 + LHFG],
        [3.0, -48.0, 6.8 + UHFL],
        [26.0, 0.0, 6.75 + LHFG],
        [26.0, 0.0, 6.75 + UHFL],
        [26.0, -20.0, 6.84 + SHFG],
        [27.5, -46.5, 6.88 + LHFG],
        [27.5, -46.5, 6.88 + UHFL],
        [54.0, 0.0, 6.44 + LHFG],
        [54.0, 0.0, 6.44 + UHFL],
        [53.0, -21.0, 6.48 + SHFG],
        [54.5, -46.5, 6.49 + LHFG],
        [54.5, -46.5, 6.49 + UHFL],
        [83.5, -27.0, 6.46 + LHFG],
        [83.5, -27.0, 6.46 + UHFL],
        [113.5, -40.0, 6.24 + LHFG],
        [113.5, -40.0, 6.24 + UHFL],
        [144.5, -54.0, 6.58 + LHFG],
        [144.5, -54.0, 6.58 + UHFL],
        [193.0, -101.0, 6.53 + LHFG],
        [193.0, -101.0, 6.53 + UHFL],
        [195.5, -126.0, 6.47 + LHFG],
        [195.5, -126.0, 6.47 + UHFL],
        [200.0, -145.5, 6.94 + LHFG],
        [200.0, -145.5, 6.94 + UHFL],
        [129.0, -179.5, 7.48 + LHFG],
        [129.0, -179.5, 7.48 + UHFL],
        [106.5, -183.0, 6.77 + LHFG],
        [106.5, -183.0, 6.77 + UHFL],
        [81.0, -182.0, 6.73 + LHFG],
        [81.0, -182.0, 6.73 + UHFL],
        [104.0, -158.5, 5.79 + SHFG],
        [123.0, -145.5, 5.9 + SHFG],
        [164.0, -142.5, 5.82 + SHFG],
        [175.0, -124.5, 6.1 + SHFG],
        [169.0, -106.5, 6.08 + SHFG],
        [136.0, -90.0, 5.98 + SHFG],
        [100.0, -95.6, 6.95 + SHFG],
        [75.0, -105.0, 6.55 + SHFG],
        [55.5, -114.0, 6.05 + LHFG],
        [55.5, -114.0, 6.05 + UHFL],
    ]
)
DEFAULT_KML_PATH = Path("~/Downloads/Post locations.kml")

POND_IMAGE_EXTENT = [0, 380, 0, 260]
POND_VIEW_XLIM = (POND_IMAGE_EXTENT[0], POND_IMAGE_EXTENT[1])
POND_VIEW_YLIM = (POND_IMAGE_EXTENT[2], POND_IMAGE_EXTENT[3])
MIC_DISPLAY_OFFSET_X = 70
MIC_DISPLAY_OFFSET_Y = 220

np = None
mpimg = None
patches = None
Figure = None
tk = None
ttk = None
messagebox = None
FigureCanvasTkAgg = None
NavigationToolbar2Tk = None
MIC_LOCS = None
MIC_DISPLAY_XY = None
MIC_LOC_LABELS = None
MIC_LOC_SOURCE = None


def parse_kml_mic_locs(kml_path: Path) -> tuple[list[list[float]], list[str]]:
    root = ET.parse(kml_path).getroot()
    namespace = {"kml": "http://www.opengis.net/kml/2.2"}
    locs = []
    labels = []

    for placemark in root.findall(".//kml:Placemark", namespace):
        name = placemark.findtext("kml:name", namespaces=namespace)
        if not name:
            continue

        try:
            loc = [float(value.strip()) for value in name.split(",")]
        except ValueError:
            continue

        if len(loc) != 3:
            continue

        locs.append(loc)
        labels.append(name)

    if not locs:
        raise ValueError(f"No x,y,z placemark names found in {kml_path}")

    return locs, labels


def import_plotting_backend(kml_path: Path | None = None) -> None:
    global Figure, MIC_DISPLAY_XY, MIC_LOC_LABELS, MIC_LOC_SOURCE, MIC_LOCS, mpimg, np, patches

    mpl_config_dir = Path(tempfile.gettempdir()) / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

    xdg_cache_dir = Path(tempfile.gettempdir()) / "fontconfig-cache"
    xdg_cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("XDG_CACHE_HOME", str(xdg_cache_dir))

    try:
        import matplotlib.image as mpimg_module
        import matplotlib.patches as patches_module
        import numpy as np_module
        from matplotlib.figure import Figure as figure_class
    except ImportError as exc:
        raise SystemExit(
            "This script needs numpy and matplotlib. Try running it from the "
            "notebook environment, for example:\n"
            "  /Users/adityakrishna/miniforge3/envs/bat_msds/bin/python "
            "scripts/carp_pond_active_space_gui.py"
        ) from exc

    np = np_module
    mpimg = mpimg_module
    patches = patches_module
    Figure = figure_class
    expanded_kml_path = kml_path.expanduser() if kml_path is not None else None
    if expanded_kml_path is not None and expanded_kml_path.exists():
        mic_locs_points, mic_loc_labels = parse_kml_mic_locs(expanded_kml_path)
        MIC_LOC_SOURCE = str(expanded_kml_path)
    else:
        mic_locs_points = FALLBACK_MIC_LOCS_POINTS
        mic_loc_labels = [f"m{index}" for index in range(len(mic_locs_points))]
        MIC_LOC_SOURCE = "fallback hard-coded points"

    MIC_LOCS = np.array(mic_locs_points, dtype=float)
    MIC_LOC_LABELS = mic_loc_labels
    MIC_DISPLAY_XY = np.column_stack(
        [
            MIC_DISPLAY_OFFSET_X + MIC_LOCS[:, 0],
            MIC_DISPLAY_OFFSET_Y + MIC_LOCS[:, 1],
        ]
    )


def import_tkinter_backend() -> None:
    global FigureCanvasTkAgg, NavigationToolbar2Tk, messagebox, tk, ttk

    try:
        import tkinter as tk_module
        from tkinter import messagebox as messagebox_module
        from tkinter import ttk as ttk_module
        from matplotlib.backends.backend_tkagg import (
            FigureCanvasTkAgg as figure_canvas_tk_agg,
        )
        from matplotlib.backends.backend_tkagg import (
            NavigationToolbar2Tk as navigation_toolbar_2_tk,
        )
    except ImportError as exc:
        raise SystemExit(
            "This script needs a Python installation with Tkinter support. "
            "Try running it from the notebook environment, for example:\n"
            "  /Users/adityakrishna/miniforge3/envs/bat_msds/bin/python "
            "scripts/carp_pond_active_space_gui.py"
        ) from exc

    tk = tk_module
    ttk = ttk_module
    messagebox = messagebox_module
    FigureCanvasTkAgg = figure_canvas_tk_agg
    NavigationToolbar2Tk = navigation_toolbar_2_tk


class CarpPondActiveSpaceGUI:
    def __init__(self, root: tk.Tk, image_path: Path) -> None:
        self.root = root
        self.root.title("Carp Pond Active Space Simulator")
        self.root.geometry("1180x760")

        self.image_path = image_path.expanduser()
        self.pond_img = self._load_pond_image(self.image_path)

        self.bird_x = tk.DoubleVar(value=160.0)
        self.bird_y = tk.DoubleVar(value=80.0)
        self.bird_z = tk.DoubleVar(value=2.0)
        self.active_radius = tk.DoubleVar(value=60.0)
        self.show_distances = tk.BooleanVar(value=True)
        self.show_inactive = tk.BooleanVar(value=True)

        self._build_layout()
        self._plot()

    def _load_pond_image(self, image_path: Path) -> np.ndarray:
        if not image_path.exists():
            messagebox.showerror(
                "Missing image",
                f"Could not find Carp Pond background image:\n{image_path}",
            )
            raise FileNotFoundError(image_path)
        return mpimg.imread(image_path)

    def _build_layout(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        control_frame = ttk.Frame(self.root, padding=12)
        control_frame.grid(row=0, column=0, sticky="ns")

        plot_frame = ttk.Frame(self.root)
        plot_frame.grid(row=0, column=1, sticky="nsew")
        plot_frame.columnconfigure(0, weight=1)
        plot_frame.rowconfigure(0, weight=1)

        self.figure = Figure(figsize=(10, 6), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame, pack_toolbar=False)
        toolbar.update()
        toolbar.grid(row=1, column=0, sticky="ew")

        ttk.Label(control_frame, text="Bird Position").grid(row=0, column=0, sticky="w")
        self._add_slider(
            control_frame,
            "x (m)",
            self.bird_x,
            POND_VIEW_XLIM[0],
            POND_VIEW_XLIM[1],
            1,
            row=1,
        )
        self._add_slider(
            control_frame,
            "y (m)",
            self.bird_y,
            POND_VIEW_YLIM[0],
            POND_VIEW_YLIM[1],
            1,
            row=2,
        )
        self._add_slider(control_frame, "z (m)", self.bird_z, 0, 50, 0.5, row=3)

        ttk.Separator(control_frame).grid(row=4, column=0, sticky="ew", pady=12)
        ttk.Label(control_frame, text="Active Space").grid(row=5, column=0, sticky="w")
        self._add_slider(control_frame, "radius (m)", self.active_radius, 5, 150, 1, row=6)

        ttk.Checkbutton(
            control_frame,
            text="Show distances",
            variable=self.show_distances,
            command=self._plot,
        ).grid(row=7, column=0, sticky="w", pady=(12, 0))
        ttk.Checkbutton(
            control_frame,
            text="Show inactive mics",
            variable=self.show_inactive,
            command=self._plot,
        ).grid(row=8, column=0, sticky="w")

        self.status_label = ttk.Label(control_frame, text="", justify="left")
        self.status_label.grid(row=9, column=0, sticky="ew", pady=(18, 0))

        ttk.Button(control_frame, text="Reset", command=self._reset).grid(
            row=10, column=0, sticky="ew", pady=(18, 0)
        )

    def _add_slider(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.DoubleVar,
        from_: float,
        to: float,
        resolution: float,
        row: int,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=6)
        frame.columnconfigure(0, weight=1)

        value_label = ttk.Label(frame, width=9)
        value_label.grid(row=0, column=1, sticky="e")

        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w")

        def update_label(*_: object) -> None:
            value_label.configure(text=f"{variable.get():.1f}")

        variable.trace_add("write", update_label)
        update_label()

        scale = ttk.Scale(
            frame,
            variable=variable,
            from_=from_,
            to=to,
            command=lambda _value: self._plot(),
        )
        scale.grid(row=1, column=0, columnspan=2, sticky="ew")

        spinbox = ttk.Spinbox(
            frame,
            textvariable=variable,
            from_=from_,
            to=to,
            increment=resolution,
            width=8,
            command=self._plot,
        )
        spinbox.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(2, 0))
        spinbox.bind("<Return>", lambda _event: self._plot())
        spinbox.bind("<FocusOut>", lambda _event: self._plot())

    def _reset(self) -> None:
        self.bird_x.set(160.0)
        self.bird_y.set(80.0)
        self.bird_z.set(2.0)
        self.active_radius.set(60.0)
        self.show_distances.set(True)
        self.show_inactive.set(True)
        self._plot()

    def _plot(self) -> None:
        bird_x = self.bird_x.get()
        bird_y = self.bird_y.get()
        bird_z = self.bird_z.get()
        active_radius = self.active_radius.get()

        bird_xyz = np.array([bird_x, bird_y, bird_z], dtype=float)
        mic_xyz = np.column_stack([MIC_DISPLAY_XY, MIC_LOCS[:, 2]])
        distances_to_mics = np.linalg.norm(mic_xyz - bird_xyz, axis=1)
        active_mic_mask = distances_to_mics <= active_radius
        active_count = int(active_mic_mask.sum())

        self.ax.clear()
        self.ax.imshow(self.pond_img, extent=POND_IMAGE_EXTENT, origin="upper", alpha=1)

        active_space = patches.Circle(
            (bird_x, bird_y),
            active_radius,
            facecolor="yellow",
            edgecolor="yellow",
            alpha=0.18,
            linewidth=2,
            label="Active-space footprint",
            zorder=1,
        )
        self.ax.add_patch(active_space)

        self.ax.scatter(
            bird_x,
            bird_y,
            s=120,
            facecolor="yellow",
            edgecolor="k",
            linewidth=2,
            label="Simulated bird position",
            zorder=4,
        )

        if self.show_inactive.get() and np.any(~active_mic_mask):
            inactive_xy = MIC_DISPLAY_XY[~active_mic_mask]
            self.ax.scatter(
                inactive_xy[:, 0],
                inactive_xy[:, 1],
                s=180,
                marker="*",
                facecolor="0.65",
                edgecolor="k",
                linewidth=1.5,
                label="Inactive microphone",
                zorder=3,
            )

        if np.any(active_mic_mask):
            active_xy = MIC_DISPLAY_XY[active_mic_mask]
            self.ax.scatter(
                active_xy[:, 0],
                active_xy[:, 1],
                s=220,
                marker="*",
                facecolor="limegreen",
                edgecolor="k",
                linewidth=2,
                label="Active microphone",
                zorder=4,
            )

        for mic_i, (recorder_display_pos, distance_to_mic, is_active) in enumerate(
            zip(MIC_DISPLAY_XY, distances_to_mics, active_mic_mask)
        ):
            if not (is_active or self.show_inactive.get()):
                continue

            line_color = "limegreen" if is_active else "0.35"
            self.ax.plot(
                [bird_x, recorder_display_pos[0]],
                [bird_y, recorder_display_pos[1]],
                color=line_color,
                linewidth=2 if is_active else 1,
                linestyle="dashed",
                alpha=0.9 if is_active else 0.45,
                zorder=2,
            )

            if self.show_distances.get():
                self.ax.text(
                    1 + ((bird_x + recorder_display_pos[0]) / 2),
                    (bird_y + recorder_display_pos[1]) / 2,
                    s=f"{MIC_LOC_LABELS[mic_i]}: {distance_to_mic:.1f} m",
                    color="w",
                    fontsize=9,
                    zorder=5,
                )

        self.ax.set_title(
            "Bird=({:.1f}, {:.1f}, {:.1f}) m | active radius={:.1f} m | "
            "active mics={}/{}".format(
                bird_x,
                bird_y,
                bird_z,
                active_radius,
                active_count,
                len(MIC_LOCS),
            )
        )
        self.ax.legend(loc="upper right", framealpha=1)
        self.ax.set_xlim(*POND_VIEW_XLIM)
        self.ax.set_ylim(*POND_VIEW_YLIM)
        self.ax.set_aspect("equal")
        self.ax.axis("off")

        self.status_label.configure(
            text=(
                f"Image: {self.image_path}\n"
                f"Active microphones: {active_count}/{len(MIC_LOCS)}\n"
                f"Mic source: {MIC_LOC_SOURCE}\n"
                f"Plan-view radius: {active_radius:.1f} m"
            )
        )
        self.canvas.draw_idle()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open a Tkinter GUI for Carp Pond bird active-space simulation."
    )
    parser.add_argument(
        "--image-path",
        type=Path,
        default=Path("~/Desktop/Screenshot 2026-05-26 at 2.39.01 PM.png"),
        help="Path to the Carp Pond aerial image.",
    )
    parser.add_argument(
        "--kml-path",
        type=Path,
        default=DEFAULT_KML_PATH,
        help="Path to the Google Earth KML containing post locations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    import_plotting_backend(args.kml_path)
    import_tkinter_backend()
    root = tk.Tk()
    CarpPondActiveSpaceGUI(root, args.image_path)
    root.mainloop()


if __name__ == "__main__":
    main()
