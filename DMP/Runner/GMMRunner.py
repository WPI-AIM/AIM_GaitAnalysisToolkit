import RunnerBase
import numpy as np


class GMMRunner(RunnerBase.RunnerBase):

    def __init__(self, file):
        super(GMMRunner, self).__init__(file)
        self._x = self.get_start()
        self._dx = np.array([[0.0]])
        self._path = []
        self._index = 0

    def step(self):
        L = np.append(np.eye(1) * 50.0, np.eye(1) * 10.0, 1)
        x_ = np.append(self.get_goal() - self._x, -self._dx).reshape((-1, 1))
        ddx = L.dot(x_) + (self._get_expData()[:, self._index] * self._get_sIn()[self._index]).reshape((-1, 1))
        self._dx = self._dx + ddx * self.get_dt()
        self._x = self._x + self._dx * self.get_dt()
        self._index = self._index + 1
        self._path.append(self._x[0])
        return self._x

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if value >= self.get_length():
            self._index = 0
        else:
            self._index = value

    def get_length(self):
        return self._data["len"]

    def get_H(self):
        return self._data["H"]

    def get_expData(self):
        return self._data["expData"]

    def get_expSigma(self):
        return self._data["expSigma"]

    def get_start(self):
        return self._data["start"]

    def get_goal(self):
        return self._data["goal"]

    def get_dt(self):
        return self._data["dt"]