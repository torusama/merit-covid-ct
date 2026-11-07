"""Main XYZ API using Duong strict Z,Y,X probability reconstruction."""
import numpy as np
from merit_covid.data.pipeline.metrics import SliceReconstructor

def reconstruct_volume(predictions, shape):
    if len(shape) != 3 or any((type(v) is not int or v < 1 for v in shape)):
        raise ValueError('Expected a positive 3D shape')
    reconstruction = SliceReconstructor((shape[2], shape[1], shape[0]))
    for index, values in predictions:
        if isinstance(index, (bool, np.bool_)):
            raise ValueError('Invalid slice index')
        reconstruction.add(index, values)
    return reconstruction.finish().transpose(2, 1, 0).copy()
