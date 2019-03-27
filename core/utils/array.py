import numpy as np


def permute_axes_subtract(arr, axis=1):
    """
            calculates all the differences between all combinations
            terms in the input array. output[i,j] = arr[i] - arr[j]
            for every combination if ij.

            Parameters
            ----------
            arr numpy.array
                a 1d input array

            Returns
            -------
            numpy.array
                a 2d array

            Examples
            --------
            arr = [10, 11, 12]

            diffs = [[ 0 -1 -2]
                    [ 1  0 -1]
                    [ 2  1  0]]
            """
    s = arr.shape
    if arr.ndim == 1:
        axis = 0

    # Get broadcastable shapes by introducing singleton dimensions
    s1 = np.insert(s, axis, 1)
    s2 = np.insert(s, axis + 1, 1)

    # Perform subtraction after reshaping input array to
    # broadcastable ones against each other
    return arr.reshape(s1) - arr.reshape(s2)
