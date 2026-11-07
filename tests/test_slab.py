import numpy as np
import pytest

from merit_covid.data.slab import sample_slab


@pytest.mark.parametrize("depth", [1, 3, 5, 11])
def test_center_order_boundary_and_physical_offsets(depth):
    image = np.broadcast_to(np.arange(15), (4, 6, 15)).copy()
    mask = (image == 7).astype(np.uint8)
    args = dict(num_slices=depth, master_index=depth // 2)
    sample = sample_slab(image, mask, 7, np.diag([0.7, 0.8, 2.5, 1]), **args)
    assert sample["image"].shape == (1, depth, 4, 6)
    assert sample["mask"].shape == (1, 4, 6)
    expected = np.arange(depth) + 7 - depth // 2
    np.testing.assert_array_equal(sample["image"][0, :, 0, 0], expected)
    np.testing.assert_allclose(sample["z_offsets_mm"], (expected - 7) * 2.5)
    edge = sample_slab(image, mask, 0, np.diag([1, 1, 5, 1]), **args)
    assert np.all(edge["z_offsets_mm"][:depth // 2 + 1] == 0)
    assert edge["context_indices"][depth // 2] == 0


def test_reflect_short_volume_and_real_positions():
    image = np.arange(2, dtype=np.float32).reshape(1, 1, 2)
    sample = sample_slab(image, None, 0, np.diag([1, 1, 6, 1]), boundary="reflect")
    np.testing.assert_array_equal(sample["context_indices"], [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    np.testing.assert_array_equal(sample["z_offsets_mm"], sample["context_indices"] * 6)
    single = sample_slab(image[:, :, :1], None, 0, np.eye(4), boundary="reflect")
    assert not single["z_offsets_mm"].any()


def test_invalid_slab():
    image = np.zeros((2, 2, 3))
    with pytest.raises(ValueError):
        sample_slab(image, None, 3, np.eye(4))
    with pytest.raises(ValueError):
        sample_slab(image, None, 1, np.eye(4), num_slices=4)
