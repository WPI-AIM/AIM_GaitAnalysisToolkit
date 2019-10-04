from collections import deque

import numpy as np

class Mean_Filter():
    """
    implementation of a mean filter
    """
    def __init__(self, size=10):
        self._sample_window = deque([], size)

    def update(self, value):
        self._sample_window.append(value)
        return np.sum(self._sample_window, 0) / len(self._sample_window)