import numpy as np
from .ModelBase import gaussPDF
from numpy import matlib

class GMR(object):

    def __init__(self, mu, sigma, priors):
        """

        :param mu: list of trained mus
        :param sigma: list of trained sigmas
        :param priors: list of trained proirs
        """
        self._sigma = sigma
        self._mu = mu
        self._priors = priors
        self._states = len(mu[0])

    @property
    def sigma(self):
        """

        :return: sigma
        """
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        """
        set the sigma value
        :param value: value to set sigma
        :return:
        """
        self._sigma = value

    @property
    def mu(self):
        """

        :return: list of mu
        """
        return self._mu

    @mu.setter
    def mu(self, value):
        """

        :param value: value to set mu
        :return:
        """
        self._mu = value

    @property
    def priors(self):
        """

        :return: return the priors
        """
        return self._priors

    @priors.setter
    def priors(self, value):
        """

        :param value: value to set the priors
        :return: None
        """
        self._priors = value

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, value):
        self._states = value

    def train(self, DataIn, in_, out_, reg=1e-8):
        """
        Train the system
        :param DataIn: data to tain on
        :param in_: start location
        :param out_: end location
        :param reg: regulization
        :return:
        """
        nbData = np.shape(DataIn)[0]
        nbVarOut = len(out_)
        in_ = in_[0]

        MuTmp = np.zeros((nbVarOut, self.states))
        expData = np.zeros((nbVarOut, nbData))
        expSigma = []
        for i in range(nbData):
            expSigma.append(np.zeros((nbVarOut, nbVarOut)))

        H = np.zeros((self.states, nbData))

        for t in range(nbData):

            for i in range(self.states):
                H[i, t] = self.priors[i] * gaussPDF(np.asarray([DataIn[t]]), self.mu[in_, i], self.sigma[i][in_, in_])

            H[:, t] = H[:, t] / np.sum(H[:, t] + np.finfo(float).tiny)

            for i in range(self.states):
                MuTmp[:, i] = self.mu[out_[0]:out_[-1]+1, i] + self.sigma[i][out_[0]:out_[-1]+1, in_] / \
                              self.sigma[i][in_, in_] * (DataIn[t] - self.mu[in_, i])

                expData[:, t] = expData[:, t] + H[i, t] * MuTmp[:, i]

            for i in range(self.states):
                sigma_tmp = self.sigma[i][out_[0]:out_[-1]+1, out_[0]:out_[-1]+1] - \
                            (self.sigma[i][out_[0]:out_[-1]+1, in_] / self.sigma[i][in_, in_]).reshape((-1,1))  * \
                             self.sigma[i][in_, out_[0]:out_[-1]+1]
                expSigma[t] = expSigma[t] + H[i, t] * (sigma_tmp + MuTmp[:, i].reshape((-1, 1)) * MuTmp[:, i].T)

            expSigma[t] = expSigma[t] - np.dot(expData[:, t].reshape((-1,1)),expData[:, t].reshape((1,-1))) + np.eye(nbVarOut) * reg[1:]

        return expData, expSigma, H
