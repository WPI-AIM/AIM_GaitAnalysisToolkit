
import numpy as np
import pickle
class RunnerBase(object):

    def __init__(self, file):


        if ".pickle" in file:
            self._file = file
        else:
            self._file = file + ".pickle"

        self._goal = None
        self._kp = 50.0
        self._kc = 10.0
        self._x = 0
        self._dx = 0
        self._ddx = 0
        self._index = 0
        self._path = []
        self._K = None

        with open(self._file, 'rb') as handle:
            self._data = pickle.load(handle)

    @property
    def x(self):
        return self._x

    @x.setter
    def x(self, value):
        self._x = value

    @property
    def dx(self):
        return self._dx

    @property
    def goals(self):
        return self._goals


    @dx.setter
    def dx(self, value):
        self._dx = value

    @property
    def ddx(self):
        return self._ddx

    @dx.setter
    def ddx(self, value):
        self._ddx = value

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

    @property
    def K(self):
        return self._K

    @K.setter
    def K(self, value):
        self._K = value


    def step(self, x=None, dx=None):
        """

        :param x: feedback position
        :param dx: feedback velocity
        :return: None
        """
        pass

    def run(self):
        path = []
        for i in range(self.get_length()):
            path.append(self.step())
        self._index = 0
        self._x = self.get_start()
        self._goal = self._data["goal"]
        self._dx = np.array([[0.0]])
        return np.array(path)

    def reset(self):
        self._index = 0
        self._x = self.get_start()
        self._goal = self._data["goal"]
        self._dx = np.zeros(len(self._x)).reshape((-1, 1))
        self._v0 = np.zeros(len(self._x)).reshape((-1, 1))  # vector of zeros for velocity


    @property
    def goal(self):
        return self._goal

    def get_H(self):
        return self._data["H"]

    def get_goals(self):
        return self._data["goals"]

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

def calculate_imitation_metric_spatially(demos, imitation):
    M = len(demos)
    T = len(imitation)
    metric = 0.0

    for d in demos:
        metric += np.sum(np.sqrt(np.power( imitation.flatten() - d.flatten(),2)))

    return metric
