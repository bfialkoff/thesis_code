from itertools import product

import numpy as np


def permute_difference(arr):
    def _permute_difference(arr):
        opts = np.array(list(product(arr, arr)))
        d = (opts[:, 0] - opts[:, 1]).reshape(len(arr), -1)
        return d

    if len(arr.shape) == 1 or arr.shape[1] == 1:
        diffs = _permute_difference(arr)
    else:
        diffs = np.apply_along_axis(permute_difference, 1, arr)
    return diffs
