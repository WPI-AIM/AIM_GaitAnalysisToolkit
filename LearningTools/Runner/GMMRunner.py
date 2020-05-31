import LearningTools.Runner.RunnerBase as RunnerBase
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

    def step(self, x=None, dx=None):
        """

        :param x: feedback position
        :param dx: feedback velocity
        :return: None
        """

        super(GMMRunner, self).step(x, dx)
        L = np.append(np.eye(1) * self._kp, np.eye(1) * self._kc, 1)
        x_ = np.append(self.goal - self._x, -self._dx).reshape((-1, 1))
        ddx = L.dot(x_) + (self.get_expData()[:, self._index] * self.get_sIn()[self._index]).reshape((-1, 1))
        self._dx = self._dx + ddx * self.get_dt()
        self._x = self._x + self._dx * self.get_dt()
        self._index = self._index + 1
        self._path.append(self._x[0])
        self._K = L
        return self._x[0]

    @property
    def goal(self):
        return self._goal

    @property
    def index(self):
        return self._index

    @index.setter
    def index(self, value):
        if value > self.get_length():
            self._index = 0
        else:
            self._index = value

    @property
    def goal(self):
        return self._goal

