"""Interactive CT slice viewer for both local COVID lesion datasets.

Run from the repository root:
    python tools/view_slices.py

To create a PNG without opening a window:
    python tools/view_slices.py --preview --dataset zenodo20
    python tools/view_slices.py --preview --dataset covid19_20
"""

from __future__ import annotations

import argparse
import json
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import nibabel as nib
import numpy as np
from PIL import Image, ImageDraw, ImageTk
from merit_covid.data.manifest import legacy_mask_path
from merit_covid.data.nifti_io import verify_alignment


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"
LUNG_COLORS = {1: (34, 197, 94), 2: (59, 130, 246)}
LESION_COLOR = np.asarray((239, 68, 68), dtype=np.float32)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    title: str
    root: Path
    has_lung_masks: bool

    @property
    def manifest(self) -> Path:
        return self.root / "cases.json"


DATASETS = {
    "zenodo20": DatasetSpec(
        "zenodo20", "Zenodo / Ma et al. (20 cases)", ROOT / "data" / "zenodo20", True
    ),
    "covid19_20": DatasetSpec(
        "covid19_20", "COVID-19-20 Challenge (199 cases)", ROOT / "data" / "covid19_20", False
    ),
}


def load_cases(spec: DatasetSpec) -> list[dict]:
    if not spec.manifest.exists():
        raise FileNotFoundError(f"Dataset manifest not found: {spec.manifest}")
    return json.loads(spec.manifest.read_text(encoding="utf-8"))


def window_ct(slice_ct: np.ndarray) -> np.ndarray:
    """Display-only heuristic; never use this function for model preprocessing."""
    if np.nanmin(slice_ct) < -300:
        low, high = -1000.0, 400.0
    else:
        low, high = 0.0, 255.0
    gray = (np.clip((slice_ct - low) / (high - low), 0, 1) * 255).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def mask_image(mask: np.ndarray, color: tuple[int, int, int]) -> np.ndarray:
    result = np.zeros((*mask.shape, 3), dtype=np.uint8)
    result[mask] = color
    return result


def render_panels(spec: DatasetSpec, loaded: dict, index: int) -> list[tuple[str, Image.Image]]:
    ct = np.rot90(np.asarray(loaded["image"].dataobj[:, :, index], dtype=np.float32))
    lesion = np.rot90(np.asarray(loaded["mask"].dataobj[:, :, index])) > 0
    base = window_ct(ct)
    overlay = base.copy()
    overlay[lesion] = (base[lesion].astype(np.float32) * 0.55 + LESION_COLOR * 0.45).astype(np.uint8)
    panels = [
        ("CT", Image.fromarray(base)),
        ("COVID lesion", Image.fromarray(mask_image(lesion, tuple(LESION_COLOR.astype(int))))),
        ("CT + lesion", Image.fromarray(overlay)),
    ]
    if spec.has_lung_masks:
        lung = np.rot90(np.asarray(loaded["lung"].dataobj[:, :, index], dtype=np.uint8))
        combined = np.rot90(np.asarray(loaded["combined"].dataobj[:, :, index], dtype=np.uint8))
        lung_rgb = np.zeros_like(base)
        combined_rgb = np.zeros_like(base)
        for label, color in LUNG_COLORS.items():
            lung_rgb[lung == label] = color
            combined_rgb[combined == label] = color
        combined_rgb[combined == 3] = LESION_COLOR.astype(np.uint8)
        panels[1:1] = [
            ("Left / right lung", Image.fromarray(lung_rgb)),
            ("Combined mask", Image.fromarray(combined_rgb)),
        ]
    return panels


def load_case_images(spec: DatasetSpec, case: dict) -> dict:
    loaded = {
        "image": nib.load(str(spec.root / case["image"])),
        "mask": nib.load(str(legacy_mask_path(case, spec.root, "mask"))),
    }
    if spec.has_lung_masks:
        loaded["lung"] = nib.load(str(legacy_mask_path(case, spec.root, "lung_mask")))
        loaded["combined"] = nib.load(str(legacy_mask_path(case, spec.root, "combined_mask")))
    for key, value in loaded.items():
        if key != "image":
            verify_alignment(loaded["image"], value)
    return loaded


def make_montage(panels: list[tuple[str, Image.Image]], case: dict, index: int) -> Image.Image:
    width, height = panels[0][1].size
    columns, gap, top = 3, 12, 32
    rows = (len(panels) + columns - 1) // columns
    output = Image.new("RGB", (columns * width + (columns + 1) * gap, rows * (height + top) + (rows + 1) * gap + 22), "#111827")
    draw = ImageDraw.Draw(output)
    for position, (name, panel) in enumerate(panels):
        x = gap + (position % columns) * (width + gap)
        y = gap + (position // columns) * (height + top + gap)
        draw.text((x + 4, y + 8), name, fill="white")
        output.paste(panel, (x, y + top))
    draw.text((gap + 4, output.height - 20), f"{case['id']} | slice {index + 1}/{case['shape'][2]}", fill="white")
    return output


class SliceViewer:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.spec = DATASETS["zenodo20"]
        self.cases: list[dict] = []
        self.case: dict | None = None
        self.loaded: dict = {}
        self.panels: list[tuple[str, Image.Image]] = []
        self.photos: list[ImageTk.PhotoImage] = []
        self.panel_frames: list[ttk.LabelFrame] = []
        self.panel_labels: list[ttk.Label] = []

        root.title("COVID CT slice viewer")
        root.geometry("1260x920")
        root.minsize(850, 700)

        controls = ttk.Frame(root, padding=12)
        controls.pack(fill="x")
        ttk.Label(controls, text="Dataset").grid(row=0, column=0, padx=(0, 8))
        self.dataset_select = ttk.Combobox(controls, state="readonly", width=34, values=[item.title for item in DATASETS.values()])
        self.dataset_select.grid(row=0, column=1, padx=(0, 20))
        self.dataset_select.current(0)
        self.dataset_select.bind("<<ComboboxSelected>>", self.change_dataset)
        ttk.Label(controls, text="CT case").grid(row=0, column=2, padx=(0, 8))
        self.case_select = ttk.Combobox(controls, state="readonly", width=36)
        self.case_select.grid(row=0, column=3, padx=(0, 20))
        self.case_select.bind("<<ComboboxSelected>>", self.change_case)
        ttk.Button(controls, text="Save PNG", command=self.save_png).grid(row=0, column=4)

        slice_bar = ttk.Frame(root, padding=(12, 0, 12, 8))
        slice_bar.pack(fill="x")
        ttk.Button(slice_bar, text="<", width=4, command=lambda: self.step(-1)).pack(side="left")
        self.slider = tk.Scale(slice_bar, from_=0, to=1, orient="horizontal", showvalue=False, command=self.draw)
        self.slider.pack(side="left", fill="x", expand=True, padx=8)
        ttk.Button(slice_bar, text=">", width=4, command=lambda: self.step(1)).pack(side="left")
        self.status = ttk.Label(slice_bar, text="")
        self.status.pack(side="left", padx=(14, 0))
        self.description = ttk.Label(root, padding=(12, 2, 12, 10), wraplength=1150)
        self.description.pack(anchor="w")

        self.grid = ttk.Frame(root, padding=(10, 0, 10, 10))
        self.grid.pack(fill="both", expand=True)
        for column in range(3):
            self.grid.columnconfigure(column, weight=1)
        for row in range(2):
            self.grid.rowconfigure(row, weight=1)
        for _ in range(5):
            frame = ttk.LabelFrame(self.grid, padding=6)
            label = ttk.Label(frame, anchor="center")
            label.pack(fill="both", expand=True)
            self.panel_frames.append(frame)
            self.panel_labels.append(label)

        root.bind("<Left>", lambda _: self.step(-1))
        root.bind("<Right>", lambda _: self.step(1))
        root.bind("<MouseWheel>", lambda event: self.step(1 if event.delta < 0 else -1))
        self.set_dataset(self.spec)

    def set_dataset(self, spec: DatasetSpec) -> None:
        self.spec = spec
        self.cases = load_cases(spec)
        self.case_select["values"] = [case["id"] for case in self.cases]
        self.case_select.current(0)
        detail = "20 cases: CT, left/right lung masks, COVID lesion mask, and combined mask." if spec.has_lung_masks else "199 cases: CT and a binary COVID lesion mask."
        self.description.configure(text=detail)
        self.change_case()

    def change_dataset(self, _event=None) -> None:
        key = list(DATASETS)[self.dataset_select.current()]
        self.set_dataset(DATASETS[key])

    def change_case(self, _event=None) -> None:
        index = self.case_select.current()
        if index < 0:
            return
        self.case = self.cases[index]
        self.loaded = load_case_images(self.spec, self.case)
        self.slider.configure(to=self.case["shape"][2] - 1)
        self.slider.set(self.case["shape"][2] // 2)
        self.draw()

    def step(self, delta: int) -> None:
        if self.case is None:
            return
        maximum = self.case["shape"][2] - 1
        self.slider.set(max(0, min(maximum, self.slider.get() + delta)))

    def draw(self, _value=None) -> None:
        if self.case is None:
            return
        index = self.slider.get()
        self.panels = render_panels(self.spec, self.loaded, index)
        self.photos = [ImageTk.PhotoImage(panel.resize((330, 330), Image.Resampling.BILINEAR)) for _, panel in self.panels]
        for position, frame in enumerate(self.panel_frames):
            if position < len(self.panels):
                frame.configure(text=self.panels[position][0])
                frame.grid(row=position // 3, column=position % 3, sticky="nsew", padx=5, pady=5)
                self.panel_labels[position].configure(image=self.photos[position])
            else:
                frame.grid_forget()
        self.status.configure(text=f"{self.case['id']} | slice {index + 1}/{self.case['shape'][2]}")

    def save_png(self) -> None:
        if self.case is None:
            return
        index = self.slider.get()
        destination = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")],
            initialfile=f"{self.spec.key}_{self.case['id']}_slice_{index + 1:03d}.png",
        )
        if destination:
            make_montage(self.panels, self.case, index).save(destination)
            messagebox.showinfo("Saved", destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preview", action="store_true", help="Save a PNG without opening the GUI")
    parser.add_argument("--dataset", choices=DATASETS, default="zenodo20")
    args = parser.parse_args()
    spec = DATASETS[args.dataset]
    if args.preview:
        case = load_cases(spec)[0]
        loaded = load_case_images(spec, case)
        index = case["shape"][2] // 2
        OUTPUTS.mkdir(exist_ok=True)
        output = OUTPUTS / f"preview_{spec.key}.png"
        make_montage(render_panels(spec, loaded, index), case, index).save(output)
        print(output)
        return
    root = tk.Tk()
    SliceViewer(root)
    root.mainloop()


if __name__ == "__main__":
    main()
