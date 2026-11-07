import numpy as np
import pytest

from merit_covid.engine.reconstruction import reconstruct_volume


def test_out_of_order_multiple_slices():
    result = reconstruct_volume([(2, np.full((3, 2), 1)), (0, np.zeros((3, 2))), (1, np.full((3, 2), 0.5))], (2, 3, 3))
    np.testing.assert_array_equal(result[0, 0], [0, 0.5, 1])


@pytest.mark.parametrize("predictions,message", [
    ([(0, np.zeros((3, 2)))], "Missing"),
    ([(0, np.zeros((3, 2))), (0, np.zeros((3, 2)))], "Duplicate"),
    ([(3, np.zeros((3, 2)))], "Invalid"),
    ([(0, np.zeros((3, 3)))], "match"),
    ([(0, np.full((3, 2), np.nan))], "finite"),
])
def test_invalid_predictions(predictions, message):
    with pytest.raises(ValueError, match=message):
        reconstruct_volume(predictions, (2, 3, 3))
