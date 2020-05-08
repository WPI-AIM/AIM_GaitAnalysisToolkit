
import numpy as np
import pickle
class RunnerBase(object):

    def __init__(self, file):

        self._file = file
        self._goal = None
        self._kp = 50.0
        self._kc = 10.0
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

    def get_H(self):
        return self._data["H"]

    def get_expData(self):
        return self._data["expData"]

    def get_expSigma(self):
        return self._data["expSigma"]

    def get_start(self):
        return self._data["start"]

    def get_dt(self):
        return self._data["dt"]

    def get_sIn(self):
        return self._data["sIn"]

    def get_mu(self):
        return self._data["mu"]

    def get_sigma(self):
        return self._data["sigma"]

    def get_tau(self):
        return self._data["tau"]

    def get_motion(self):
        return self._data["motion"]

    def get_dwt(self):
        return self._data["dtw"]

    def get_length(self):
        return self._data["len"]

    @property
    def start(self):
        return self._x

    def update_goal(self, value):
        self._goal = value

    def update_start(self, value):
        self._x = value