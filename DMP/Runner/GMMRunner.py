import RunnerBase
import numpy as np


class GMMRunner(RunnerBase.RunnerBase):

    def __init__(self, file):
        super(GMMRunner, self).__init__(file)
        self._x = self.get_start()
        self._goal = self._data["goal"]
        self._dx = np.array([[0.0]])
        self._path = []
        self._index = 0
        self._kp = 50.0
        self._kc = 10.0

    def step(self):
        L = np.append(np.eye(1) * self._kp, np.eye(1) * self._kc, 1)
        x_ = np.append(self.goal - self._x, -self._dx).reshape((-1, 1))
        ddx = L.dot(x_) + (self.get_expData()[:, self._index] * self.get_sIn()[self._index]).reshape((-1, 1))
        self._dx = self._dx + ddx * self.get_dt()
        self._x = self._x + self._dx * self.get_dt()
        self._index = self._index + 1
        self._path.append(self._x[0])
        return self._x[0]

    def run(self):
        path = []
        for i in xrange(self.get_length()):
            path.append(self.step())
        return path

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

    @property
    def goal(self):
        return self._goal

    def get_dt(self):
        return self._data["dt"]

    def get_sIn(self):
        return self._data["sIn"]