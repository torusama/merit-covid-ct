import importlib
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from merit_covid.config import Config, load_config
from merit_covid.models.base import master_slice_input


def test_config_paths_and_validation(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("dataset_root: data\noutput_dir: results\nnum_slices: 3\nmaster_index: 1\nimage_size: [8, 9]\n")
    config = load_config(path)
    assert config.dataset_root == tmp_path / "data"
    assert config.image_size == (8, 9)
    with pytest.raises(ValueError):
        Config(num_slices=4)
    with pytest.raises(ValueError):
        Config(normalization="guess")


def test_master_selection():
    image = np.arange(2 * 1 * 3 * 4 * 5).reshape(2, 1, 3, 4, 5)
    selected = master_slice_input(dict(image=image, master_index=[0, 2]))
    assert selected.shape == (2, 1, 4, 5)
    np.testing.assert_array_equal(selected[1], image[1, :, 2])


def test_downloader_import_and_help():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run([sys.executable, "-m", "tools.download_datasets", "--help"], cwd=root, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert "--dataset" in result.stdout


def test_viewer_manifest_paths_and_synthetic_panels(nifti_case, tmp_path):
    pytest.importorskip("PIL")
    pytest.importorskip("tkinter")
    # Import without opening a window or reading any real external masks.
    import importlib.util
    path = Path(__file__).resolve().parents[1] / "tools" / "view_slices.py"
    spec = importlib.util.spec_from_file_location("viewer_test", path)
    viewer = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = viewer
    spec.loader.exec_module(viewer)
    case, _, _ = nifti_case()
    row = {"id": case.case_id, "image": str(case.image), "mask": str(case.infection_mask), "shape": [6, 8, 4]}
    dataset = viewer.DatasetSpec("synthetic", "Synthetic CT", tmp_path, False)
    loaded = viewer.load_case_images(dataset, row)
    panels = viewer.render_panels(dataset, loaded, 1)
    assert len(panels) == 3
    montage = viewer.make_montage(panels, row, 1)
    assert montage.width > 0
