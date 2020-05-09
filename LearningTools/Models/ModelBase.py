from termcolor import colored
import numpy as np
import copy
import matplotlib
from matplotlib.patches import Polygon
import abc

from matplotlib.collections import PatchCollection


class ModelBase(object):

    def __init__(self, nb_states, nb_dim=2, init_zeros=False, mu=None, lmbda=None, sigma=None, priors=None):

        # gmm.GMM.__init__(self, nb_states=nb_states, nb_dim=nb_dim)
        self._nb_dim = nb_dim
        self._nb_states = nb_states
        # flag to indicate that publishing was not init
        self.publish_init = False
        self._mu = np.zeros((self._nb_dim, self._nb_states))
        # self._lmbda = lmbda
        self._sigma = np.array([np.eye(self.nb_dim) for i in range(self.nb_states)])
        self._priors = np.ones(self.nb_states) / self.nb_states
        # self._nbData = 0
        self._reg = 1e-8
        self._nbData = 0

        if init_zeros:
            self.init_zeros()

    @abc.abstractmethod
    def init_params(self, data):
        pass

    @abc.abstractmethod
    def train(self, data, reg=1e-8, maxiter=2000):
        pass

    @abc.abstractmethod
    def get_model(self):
        pass

    @property
    def nb_dim(self):
        return self._nb_dim

    @nb_dim.setter
    def nb_dim(self, value):
        self._nb_dim = value

    @property
    def nb_states(self):
        return self._nb_states

    @nb_states.setter
    def nb_states(self, value):
        self._nb_states = value

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        self._nb_dim = value.shape[0]
        self._nb_states = value.shape[1]
        self._mu = value

    @property
    def priors(self):
        return self._priors

    @priors.setter
    def priors(self, value):
        self._priors = value

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def nbData(self):
        return self._nbData

    @nbData.setter
    def nbData(self, value):
        self._nbData = value

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, value):
        self._reg = value

    def BIC_score(self, LL):
        """
        calculate the BIC score
        :param LL:
        :return:
        """
        N = self._nbData
        D = self.nb_dim
        K = self.nb_states
        n_p = (K - 1) + K * (D + 0.5 * D * (D + 1))
        return -LL + 0.5 * n_p * np.log(N)

    def AIC_score(self, LL):
        """
        calculate the BIC score
        :param LL:
        :return:
        """
        N = self._nbData
        D = self.nb_dim
        K = self.nb_states
        n_p = (K - 1) + K * (D + 0.5 * D * (D + 1))
        return -np.log(np.power(LL,2)) + 2*K


def plot_activation(sIn, H, ax):
    nbDrawingSeg = 50
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
    nb_states = len(H)
    patches = []
    sIn_ = sIn
    sIn_.insert(0, 0)
    sIn_.append(0)

    for i in xrange(4):
        h = H[i].tolist()
        h.insert(0, 0)
        h.append(0)
        fn = map(list, zip(*[sIn_, h]))
        # patches.append(Polygon(fn,fill=None, edgecolor='r'))
        ax.add_patch(Polygon(fn, fill=None, edgecolor='r'))

# p = PatchCollection(patches)
# ax.add_collection(p)
