import nibabel as nib
import numpy as np
import pytest

from ct_pipeline.dataset import PreprocessConfig, read_volume
from ct_pipeline.metrics import (SliceReconstructor, VolumeEvaluator, aggregate_results,
                                 volume_metrics)


@pytest.mark.parametrize("pred,truth,expected", [
    ([0, 0], [0, 0], (1, 1, 1, 1)),
    ([1, 0], [0, 0], (0, 0, 0, 0)),
    ([0, 0], [1, 0], (0, 0, 0, 0)),
    ([1, 0], [1, 0], (1, 1, 1, 1)),
    ([1, 0], [0, 1], (0, 0, 0, 0)),
    ([1, 1, 0, 0], [1, 0, 1, 0], (0.5, 1 / 3, 0.5, 0.5)),
    ([1, 1, 1, 0], [1, 1, 0, 0], (0.8, 2 / 3, 2 / 3, 1)),
])
def test_manual_overlap_cases(pred, truth, expected):
    pred, truth = np.asarray(pred).reshape(1, 1, -1), np.asarray(truth).reshape(1, 1, -1)
    result = volume_metrics(pred, truth)
    np.testing.assert_allclose([result[m] for m in ("dice", "iou", "precision", "recall")], expected)


def test_threshold_shape_labels_and_finiteness():
    gt = np.array([1, 0, 1]).reshape(1, 1, 3)
    result = volume_metrics(np.array([0.5, 0.49, 1]).reshape(gt.shape), gt)
    assert result["dice"] == 1
    for pred, truth in ((np.zeros((1, 1, 2)), gt), (np.ones(gt.shape) * np.nan, gt),
                        (np.ones(gt.shape) * 1.1, gt), (gt, gt * 2), (gt[0], gt[0])):
        with pytest.raises(ValueError):
            volume_metrics(pred, truth)
    for threshold in (0, 1, np.nan):
        with pytest.raises(ValueError):
            volume_metrics(gt, gt, threshold=threshold)


def test_volume_dice_is_not_mean_slice_dice():
    gt = np.ones((1, 5, 2), dtype=np.uint8)
    gt[:, 1:, 0] = 0
    pred = gt.copy()
    pred[:, :, 1] = 0
    assert volume_metrics(pred, gt)["dice"] == pytest.approx(2 / 7)
    assert np.mean([volume_metrics(pred[:, :, i:i+1], gt[:, :, i:i+1])["dice"] for i in range(2)]) == 0.5


def test_reconstruction_order_and_errors():
    recon = SliceReconstructor((3, 2, 4))
    for index in (2, 0, 1):
        recon.add(index, np.full((1, 2, 4), index / 2))
    result = recon.finish()
    np.testing.assert_array_equal(result[:, 0, 0], [0, 0.5, 1])
    with pytest.raises(ValueError, match="Duplicate"):
        recon.add(0, np.zeros((2, 4)))
    for index in (-1, 3, 0.5):
        with pytest.raises(ValueError, match="Invalid"):
            recon.add(index, np.zeros((2, 4)))
    with pytest.raises(ValueError, match="Missing"):
        SliceReconstructor((2, 2, 4)).finish()
    with pytest.raises(ValueError, match="shape"):
        SliceReconstructor((2, 2, 4)).add(0, np.zeros((2, 3)))


def test_hd95_and_physical_lesion_volume():
    pred = np.zeros((3, 3, 3), dtype=np.uint8)
    gt = pred.copy()
    pred[1, 1, 0], gt[1, 1, 1] = 1, 1
    affine = np.diag([1.0, 2.0, 5.0, 1.0])
    result = volume_metrics(pred, gt, affine_mm=affine, include_hd95=True)
    assert result["hd95_mm"] == pytest.approx(5)
    assert result["prediction_lesion_ml"] == pytest.approx(0.01)
    assert result["lesion_volume_error_ml"] == 0
    assert volume_metrics(gt, gt, affine_mm=affine, include_hd95=True)["hd95_mm"] == 0
    pred[1, 1, 1] = 1
    result = volume_metrics(pred, gt, affine_mm=affine)
    assert result["absolute_lesion_volume_error_ml"] == pytest.approx(0.01)
    assert result["relative_lesion_volume_error"] == 1


@pytest.mark.parametrize("pred_nonempty,gt_nonempty", [(False, False), (True, False), (False, True)])
def test_hd95_empty_is_undefined(pred_nonempty, gt_nonempty):
    pred, gt = np.zeros((2, 2, 2)), np.zeros((2, 2, 2))
    pred[0, 0, 0], gt[0, 0, 0] = pred_nonempty, gt_nonempty
    result = volume_metrics(pred, gt, affine_mm=np.eye(4), include_hd95=True)
    assert result["hd95_mm"] is None
    assert result["hd95_status"] == "undefined_empty_mask"


def test_macro_equal_patient_weight_and_null_counts():
    rows = [
        {"case_id": "a1", "group_id": "a", "dice": 1, "hd95_mm": None},
        {"case_id": "a2", "group_id": "a", "dice": 0, "hd95_mm": 2},
        {"case_id": "b1", "group_id": "b", "dice": 0, "hd95_mm": None},
    ]
    report = aggregate_results(rows)
    assert report["macro"]["dice"]["mean"] == 0.25
    assert report["macro"]["dice"]["std"] == 0.25
    assert report["macro"]["hd95_mm"] == {"mean": 2, "std": 0, "valid_groups": 1, "undefined_groups": 1}
    assert report["per_group"][0]["valid_case_counts"]["hd95_mm"] == 1
    assert not report["identity"]["patient_ids_available"]
    with pytest.raises(ValueError, match="unique"):
        aggregate_results(rows + [rows[0]])


def test_evaluator_native_grid_end_to_end(write_case):
    case = write_case(affine=np.diag([-1.0, -2.0, 5.0, 1.0]))
    config = PreprocessConfig(size_hw=(12, 16))
    _, labels, _ = read_volume(case, config)
    evaluator = VolumeEvaluator(config, include_hd95=True)
    # Deliberately reversed master order; synthetic unit test, no model evaluation.
    result = evaluator.evaluate_case(case, ((i, labels[i]) for i in reversed(range(len(labels)))))
    assert result["dice"] == result["iou"] == 1
    assert result["hd95_mm"] == 0
    assert result["ground_truth_foreground_voxels"] == int(nib.load(case.mask).get_fdata().sum())
    assert evaluator.report()["macro"]["dice"]["mean"] == 1
    with pytest.raises(ValueError, match="already"):
        evaluator.evaluate_case(case, [])
