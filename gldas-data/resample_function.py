"""
array[::2] "resamples" an array taking every other point

how this works:
1. get a list of all the indices in an array on both axes
2. resample both to every other point
3. iterate over both at the same time
4. for each combo, average the point, plus the 3 cells 1 step forward in each axis
5. put that in a new "row"
6. After finishing iterating across one of the directions, put that row into the new array
7. After finishing all iterations, create an np.array with the collection

Example:
    test = np.array([i for i in range(1, 201)]).reshape((20, 10))
    print(test)
    print(resample_array(test))

"""

import numpy as np


def resample_array(a: np.array):
    new_array = []
    for i in np.array(range(a.shape[0]))[::2]:
        new_row = []
        for j in np.array(range(a.shape[1]))[::2]:
            avg = (a[i, j] + a[i + 1, j] + a[i + 1, j + 1] + a[i, j + 1]) / 4
            new_row.append(avg)
        new_array.append(new_row)
    return np.array(new_array)
