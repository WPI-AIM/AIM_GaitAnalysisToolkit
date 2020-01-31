
import numpy as np
import pickle
class RunnerBase(object):

    def __init__(self, file):

        self._file = file
        self._goal = None
        with open(self._file, 'rb') as handle:
            self._data = pickle.load(handle)


    @property
    def kp(self):
        return self._kp

    @property
    def kc(self):
        return self._kc

    @kp.setter
    def kp(self, value):
        self._kp = value

    @property
    def kc(self, value):
        self._kc = value

    def step(self):
        pass

    def run(self):
        pass

    @property
    def goal(self):
        return self._goal

    def update_goal(self, value):
        self._goal = value
