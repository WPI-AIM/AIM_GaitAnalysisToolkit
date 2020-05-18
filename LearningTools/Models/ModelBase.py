from termcolor import colored
import numpy as np
import copy
import matplotlib
from matplotlib.patches import Polygon
import abc

from matplotlib.collections import PatchCollection


class ModelBase(object):

    def __init__(self, nb_states, nb_dim=2):

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

    @abc.abstractmethod
    def init_params(self, data):
        pass

    @abc.abstractmethod
    def train(self, data, reg=1e-8, maxiter=2000):
        pass

    @abc.abstractmethod
    def get_model(self):
        """
        Get all the generated model parameters
        :return:
        """
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


def gaussPDF(x, mean, covar):
    '''Multi-variate normal distribution

    x: [n_data x n_vars] matrix of data_points for which to evaluate
    mean: [n_vars] vector representing the mean of the distribution
    covar: [n_vars x n_vars] matrix representing the covariance of the distribution

    '''

    # Check dimensions of covariance matrix:
    if type(covar) is np.ndarray:
        n_vars = covar.shape[0]
    else:
        n_vars = 1

    # Check dimensions of data:
    if x.ndim > 1 and n_vars == len(x):
        nbData = x.shape[1]
    else:
        nbData = x.shape[0]

    # nbData = x.shape[1]
    mu = np.matlib.repmat(mean.reshape((-1, 1)), 1, nbData)
    diff = (x - mu)

    # Distinguish between multi and single variate distribution:
    if n_vars > 1:
        lambdadiff = np.linalg.pinv(covar).dot(diff) * diff
        scale = np.sqrt(
            np.power((2 * np.pi), n_vars) * (abs(np.linalg.det(covar)) + 2.2251e-308))
        p = np.sum(lambdadiff, 0)
    else:
        lambdadiff = diff / covar
        scale = np.sqrt(np.power((2 * np.pi), n_vars) * covar + 2.2251e-308)
        p = diff * lambdadiff

    prop = np.exp(-0.5 * p) / scale
    return prop.T


def solve_riccati(expSigma, dt=0.01, reg =1e-8):
    ric = {}
    size = expSigma[0].shape[0]
    Ad = np.kron([[0, 1],[0, 0]], np.eye(size))*dt + np.eye(2*size)
    Q = np.zeros((size*2, size*2))
    Bd = np.kron([[0],[1]], np.eye(size))*dt
    R = np.eye(size)*reg
    P = [np.zeros((size*2, size*2))] * len(expSigma)
    P[-1][:size, :size] = np.linalg.pinv(expSigma[-1])

    for ii in xrange(len(expSigma)-2, -1, -1):
        Q[:size, :size] = np.linalg.pinv(expSigma[ii])
        B = P[ii + 1].dot(Bd)
        C = np.linalg.pinv(np.dot(Bd.T.dot(P[ii + 1]) , Bd) + R)
        D = Bd.T.dot(P[ii + 1])
        F = np.dot(np.dot(Ad.T, np.dot( np.dot(B, C), D) - P[ii + 1]), Ad)
        P[ii] = Q - F

    ric["Ad"] = Ad
    ric["Bd"] = Bd
    ric["R"] = R
    ric["P"] = P
    return ric