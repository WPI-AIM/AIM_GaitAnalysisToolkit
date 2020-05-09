import numpy as np
from ModelBase import gaussPDF

class GMR(object):

    def __init__(self, mu, sigma, priors):

        self._sigma = sigma
        self._mu = mu
        self._priors = priors
        self._states = len(mu)

    @property
    def sigma(self):
        return self._sigma

    @sigma.setter
    def sigma(self, value):
        self._sigma = value

    @property
    def mu(self):
        return self._sigma

    @mu.setter
    def mu(self, value):
        self._mu = value

    @property
    def priors(self):
        return self._priors

    @sigma.setter
    def priors(self, value):
        self._priors = value

    @property
    def states(self):
        return self._states

    @states.setter
    def states(self, value):
        self._states = value

    def train(self, DataIn, in_, out_):

        nbData = np.shape(DataIn)[0]
        nbVarOut = len(out_)
        in_ = in_[0]

        MuTmp = np.zeros((nbVarOut, self.nb_states))
        expData = np.zeros((nbVarOut, nbData))
        expSigma = []
        for i in xrange(nbData):
            expSigma.append(np.zeros((nbVarOut, nbVarOut)))

        H = np.zeros((self.states, nbData))

        for t in xrange(nbData):

            for i in xrange(self.states):
                H[i, t] = self.priors[i] * gaussPDF(np.asarray([DataIn[t]]),
                                                         self.mu[in_][i],
                                                         self.sigma[i][in_, in_])

            H[:, t] = H[:, t] / np.sum(H[:, t] + np.finfo(float).tiny)

            for i in xrange(self.states):
                MuTmp[:, i] = self.mu[out_, i] + self.sigma[i][out_, in_] / \
                              self.sigma[i][in_, in_] * \
                              (DataIn[t] - self.mu[in_, i])

                expData[:, t] = expData[:, t] + H[i, t] * MuTmp[:, i]

            for i in xrange(self.states):
                sigma_tmp = self.sigma[i][out_[0]:(out_[-1] + 1), out_[0]:(out_[-1] + 1)] - \
                            self.sigma[i][out_, in_] / self.sigma[i][in_, in_] * self.sigma[i][in_, out_]

                expSigma[t] = expSigma[t] + H[i, t] * (sigma_tmp + MuTmp[:, i].reshape((-1, 1)) * MuTmp[:, i].T)

            expSigma[t] = expSigma[t] - expData[:, t] * expData[:, t].T + np.eye(nbVarOut) * 1E-8

        return expData, expSigma, H