import numpy as np
import pytest

from merit_covid.engine.reconstruction import reconstruct_volume


def test_out_of_order_multiple_slices():
    result = reconstruct_volume([(2, np.full((2, 3), 2)), (0, np.zeros((2, 3))), (1, np.ones((2, 3)))], (2, 3, 3))
    np.testing.assert_array_equal(result[0, 0], [0, 1, 2])


@pytest.mark.parametrize("predictions,message", [
    ([(0, np.zeros((2, 3)))], "Missing"),
    ([(0, np.zeros((2, 3))), (0, np.zeros((2, 3)))], "Duplicate"),
    ([(3, np.zeros((2, 3)))], "Invalid"),
    ([(0, np.zeros((3, 3)))], "match"),
    ([(0, np.full((2, 3), np.nan))], "finite"),
])
def test_invalid_predictions(predictions, message):
    with pytest.raises(ValueError, match=message):
        reconstruct_volume(predictions, (2, 3, 3))
