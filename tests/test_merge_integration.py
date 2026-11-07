"""Cross-API regressions for main layout + Duong data semantics; synthetic only."""
from dataclasses import replace
import numpy as np
import pytest
from merit_covid.config import Config
from merit_covid.data.dataset import SlabDataset, make_dataloader
from merit_covid.data.manifest import write_manifest
from merit_covid.data.preprocessing import restore_prediction
from merit_covid.engine.reconstruction import reconstruct_volume
from merit_covid.data.pipeline import PreprocessConfig, CTSlabDataset
from merit_covid.data.pipeline.cases import Case, load_cases
from merit_covid.data.pipeline.splits import create_splits
from merit_covid.models.base import master_slice_input

@pytest.mark.parametrize("sign", [1, -1])
def test_main_and_duong_apis_share_native_non_square_data(nifti_case, sign):
    case, raw, mask = nifti_case(shape=(7, 9, 4), affine=np.diag([-0.7, -0.8, sign*5, 1]))
    main = SlabDataset([case], Config(), mode="validation")
    legacy = CTSlabDataset([Case(case.case_id, case.image, case.infection_mask, case.case_id)], PreprocessConfig())
    samples = []
    for z in range(len(main)):
        new, old = main[z], legacy[z]
        for a, b in (("image", "image_slab"), ("mask", "master_mask"), ("context_indices", "slice_indices"), ("z_offsets_mm", "z_offsets_mm")):
            np.testing.assert_array_equal(new[a], old[b])
        assert new["master_index"] == 5 and new["slice_index"] == z
        assert new["image"].shape == (1, 11, 9, 7)
        samples.append((z, new["mask"][0]))
    prepared = main.prepared_volume(0)
    reconstructed = reconstruct_volume(reversed(samples), prepared.image.shape)
    np.testing.assert_array_equal(restore_prediction(reconstructed, prepared), mask)
    assert main.__getstate__()["_cache"] == {}

@pytest.mark.parametrize("workers", [0, 1])
def test_main_loader_workers_on_shared_core(nifti_case, workers):
    case, _, _ = nifti_case()
    loader = make_dataloader(SlabDataset([case], Config(batch_size=2), mode="validation"), num_workers=workers)
    batch = next(iter(loader))
    assert tuple(batch["image"].shape) == (2, 1, 11, 8, 6)
    assert tuple(batch["mask"].shape) == (2, 1, 8, 6)

def test_versioned_main_manifest_can_feed_duong_split_loader(nifti_case, tmp_path):
    cases = [replace(nifti_case(case_id=f"v{i}")[0], patient_id=f"p{i//2}") for i in range(10)]
    path = tmp_path / "manifest.json"
    write_manifest(cases, path)
    pipeline_cases, identity = load_cases(path)
    assert identity["patient_ids_available"]
    split = create_splits(pipeline_cases, identity)
    for fold in split["folds"]:
        groups = [set(fold[role]["group_ids"]) for role in ("train", "validation", "test")]
        assert not groups[0] & groups[1] and not groups[1] & groups[2] and not groups[0] & groups[2]

def test_main_model_adapter_accepts_duong_keys_without_confusing_volume_index():
    image = np.arange(2 * 1 * 11 * 7 * 9).reshape(2, 1, 11, 7, 9)
    batch = {"image_slab": image, "master_slice_index": [100, 200]}
    np.testing.assert_array_equal(master_slice_input(batch), image[:, :, 5])
