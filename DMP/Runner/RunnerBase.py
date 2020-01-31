
import numpy as np
import pickle
class RunnerBase(object):

    def __init__(self, file):

        self._file = file

        with open(self._file, 'rb') as handle:
            self._data = pickle.load(handle)

    def step(self):
        pass

    def run(self):
        pass



