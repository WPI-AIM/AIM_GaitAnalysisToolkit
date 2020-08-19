from . import RunnerBase
import numpy as np
from random import uniform
from numpy import matlib

class TPGMMRunner(RunnerBase.RunnerBase):

    def __init__(self, file):
        """
        Running the model
        :param file: Training file
        """
        super(TPGMMRunner, self).__init__(file)
        self._x = self.get_start() # initial starting position
        self._goal = self._data["goal"] # initial ending position
        self._dx = np.zeros(len(self._x)).reshape((-1, 1))
        self._v0 = np.zeros(len(self._x)).reshape((-1, 1)) # vector of zeros for velocity
        self._path = []
        self._index = 0
        self._K = []

    def step(self, x=None, dx=None):
        """

        :param x: feedback position
        :param dx: feedback velocity
        :return: None
        """

        # allow for feedback
        if x is not None:
            self._x = x
        if dx is not None:
            self._dx = dx

        B = self.get_Bd()
        A = self.get_Ad()
        R = self.get_R()
        P = self.get_P()
        # get the gain
        v = np.linalg.inv(np.dot(np.dot(B.T, P[self._index]), B) + R)
        K = np.dot(np.dot(v.dot(B.T), P[self._index]), A)
        #K = np.append(np.eye(3) * self._kp, np.eye(3) * self._kc, 1)
        x_ = np.append(self._x, self._dx).reshape((-1, 1))
        self._ddx = K.dot(np.vstack((self.get_expData()[:, self._index].reshape((-1,1)), self._v0)) - x_)
        self._dx = self._dx + self._ddx * self.get_dt()
        self._x = self._x + self._dx * self.get_dt()
        self._index = self._index + 1
        self._path.append(self._x[0])
        self._K = K
        return self._x

    def get_Bd(self):
        return self._data["Bd"]

    def get_Ad(self):
        return self._data["Ad"]

    def get_P(self):
        return self._data["P"]

    def get_R(self):
        return self._data["R"]

    def get_K(self):
        return self._data["K"]