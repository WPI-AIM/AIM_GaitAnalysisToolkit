import RunnerBase
import numpy as np
from random import uniform

class TPGMMRunner(RunnerBase.RunnerBase):

    def __init__(self, file):
        super(TPGMMRunner, self).__init__(file)
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

        super(TPGMMRunner, self).step(x,dx)
        B = self.get_Bd()
        A = self.get_Ad()
        R = self.get_R()
        P = self.get_P()
        v = np.linalg.inv(np.dot(np.dot(B.T, P[self._index]), B) + R)
        K = np.dot(np.dot(v * B.T, P[self._index]), A)
        x_ = np.append(self._x, self._dx).reshape((-1, 1))
        ddx = K.dot(np.vstack((self.get_expData()[:, self._index], [0])) - x_)
        self._dx = self._dx + ddx * self.get_dt()
        self._x = self._x + self._dx * self.get_dt()
        self._index = self._index + 1
        self._path.append(self._x[0])
        self._K = K
        return self._x[0]

    def get_Bd(self):
        return self._data["Bd"]

    def get_Ad(self):
        return self._data["Ad"]

    def get_P(self):
        return self._data["P"]

    def get_R(self):
        return self._data["R"]