import numpy as np
import pytest

from merit_covid.engine.metrics import aggregate_patients, binary_metrics, evaluate_patients


@pytest.mark.parametrize("shape", [(2, 3), (2, 3, 4)])
def test_perfect_and_single_pixel(shape):
    mask = np.zeros(shape, dtype=np.uint8)
    mask.flat[0] = 1
    assert binary_metrics(mask, mask) == dict(dice=1, iou=1, sensitivity=1, precision=1)


def test_wrong_and_partial():
    assert binary_metrics(np.array([1, 0]), np.array([0, 1])) == dict(dice=0, iou=0, sensitivity=0, precision=0)
    metrics = binary_metrics(np.array([1, 1, 0, 0]), np.array([0, 1, 1, 0]))
    assert metrics["dice"] == 0.5
    assert metrics["iou"] == pytest.approx(1 / 3)
    assert metrics["sensitivity"] == metrics["precision"] == 0.5


def test_empty_semantics_and_patient_macro():
    empty, positive = np.zeros((2, 3), np.uint8), np.ones((2, 3), np.uint8)
    perfect = binary_metrics(empty, empty)
    assert perfect == dict(dice=1, iou=1, sensitivity=1, precision=1)
    missed = binary_metrics(empty, positive)
    assert missed == dict(dice=0, iou=0, sensitivity=0, precision=0)
    false_positive = binary_metrics(positive, empty)
    assert false_positive == dict(dice=0, iou=0, sensitivity=0, precision=0)
    assert aggregate_patients({"patient-a": perfect, "patient-b": missed})["dice"] == 0.5


def test_invalid_metric_input():
    with pytest.raises(ValueError):
        binary_metrics(np.zeros(2), np.zeros(3))
    with pytest.raises(ValueError):
        binary_metrics(np.array([0.5]), np.array([1]))


def test_linked_patient_volumes_mean_before_macro():
    one = np.ones((1, 1, 1), dtype=np.uint8)
    zero = np.zeros_like(one)
    result = evaluate_patients({"a": one, "b": zero, "c": one},
                               {"a": one, "b": one, "c": one},
                               {"a": "verified-p", "b": "verified-p", "c": None})
    assert result["per_patient"]["patient:verified-p"]["dice"] == pytest.approx(0.5)
    assert result["macro"]["dice"] == pytest.approx(0.75)
    with pytest.raises(ValueError, match="IDs"):
        evaluate_patients({"a": one}, {"b": one})
