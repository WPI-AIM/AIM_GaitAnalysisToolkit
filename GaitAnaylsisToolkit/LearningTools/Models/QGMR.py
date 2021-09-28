from . import GMR
import pyquaternion as pyq


class QGMR(GMR.GMR):

    def __init__(self, mu, sigma, priors):
        super().__init__(mu, sigma, priors)

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
                H[i, t] = self.priors[i] * gaussPDF(np.asarray([DataIn[t]]), self.mu[in_, i], self.sigma[i][in_, in_], displacement=self.displacement)

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

    def displacement(self, p1, p2):
        '''
        get the displacement between two points
        :param p1: pyquaternion
        :param p2: pyquaternion
        :return:
        '''
        return pyq.Quaternion.distance(p1, p2.conjugate())