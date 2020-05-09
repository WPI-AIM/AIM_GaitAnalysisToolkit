from termcolor import colored
import numpy as np
import copy
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import ModelBase

class TPGMM(ModelBase.ModelBase):

    def __init__(self, nb_states, nb_dim=3, init_zeros=False, mu=None, lmbda=None, sigma=None, priors=None):

        super(TPGMM, self).__init__(nb_states, nb_dim, init_zeros, mu, lmbda, sigma, priors)
        self.frames = 1

    def init_params(self, data):

        idList = self.kmeansclustering(data)
        priors = np.ones(self.nb_states) / self.nb_states
        self.sigma = np.array([np.eye(self.nb_dim) for i in range(self.nb_states)])
        self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

        for i in xrange(self.nb_states):

            idtmp = np.where(idList == i)
            mat = np.vstack((data[:, idtmp][0][0], data[:, idtmp][1][0]))

            for j in xrange(2, len(data[:, idtmp])):
                mat = np.vstack((mat, data[:, idtmp][j][0]))

            mat = np.concatenate((mat, mat), axis=1)
            priors[i] = len(idtmp[0])
            self.sigma[i] = np.cov(mat) + np.eye(self.nb_dim) * self.reg

        self.priors = priors / np.sum(priors)

    def train(self, data, reg=1e-8, maxiter=2000):

        gamma, BIC = self.em(data, reg, maxiter)
        return gamma, BIC

    def get_model(self):
        return self.sigma, self.mu

    def kmeansclustering(self, data, reg=1e-8):

        self.reg = reg

        # Criterion to stop the EM iterative update
        cumdist_threshold = 1e-10
        maxIter = 200
        minIter = 20

        # Initialization of the parameters
        cumdist_old = -1.7977e+308
        nb_step = 0
        self.nbData = data.shape[1]
        id_tmp = np.random.permutation(self.nbData)

        Mu = copy.deepcopy(data[:, id_tmp[:self.nb_states]])
        searching = True
        distTmpTrans = np.zeros((len(data[0]), self.nb_states,))
        idList = []

        while searching:

            # E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
            for i in xrange(0, self.nb_states):
                # Compute distances
                thing = np.matlib.repmat(Mu[:, i].reshape((-1, 1)), 1, self.nbData)
                temp = np.power(data - thing, 2.0)
                temp2 = np.sum(temp, 0)
                distTmpTrans[:, i] = temp2

            vTmp = np.min(distTmpTrans, 1)
            cumdist = sum(vTmp)
            idList = []

            for row, min_num in zip(distTmpTrans, vTmp):
                index = np.where(row == min_num)[0]
                idList.append(index[0])

            idList = np.array(idList)
            # M-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
            for i in xrange(self.nb_states):
                # Update the centers
                id = np.where(idList == i)
                Mu[:, i] = np.mean(data[:, id], 2).reshape((1, -1))

            # Stopping criterion %%%%%%%%%%%%%%%%%%%%
            if abs(cumdist - cumdist_old) < cumdist_threshold and nb_step > minIter:
                print 'Maximum number of kmeans iterations, ' + str(abs(cumdist - cumdist_old)) + ' is reached'
                print 'steps reached, ' + str(nb_step) + ' is reached'
                searching = False

            cumdist_old = cumdist
            nb_step = nb_step + 1

            if nb_step > maxIter:
                print 'steps reached, ' + str(nb_step) + ' is reached'
                searching = False
            print "maxitter ", nb_step

        self.mu = Mu

        return idList



    def em(self, data, reg=1e-8, maxiter=2000):

        self.reg = reg

        nb_min_steps = 50  # min num iterations
        nb_max_steps = maxiter  # max iterations
        nb_samples = data.shape[1]

        data = data.T
        searching = True
        LL = np.zeros(nb_max_steps)
        it = 0
        GAMMA = None
        while searching:

            # E - step
            L = np.zeros((self.nb_states, nb_samples))

            for i in range(self.nb_states):
                L[i, :] = self.priors[i] * self.gaussPDF(data.T, self.mu[:, i], self.sigma[i])

            GAMMA = L / np.sum(L, axis=0)
            GAMMA2 = GAMMA / np.sum(GAMMA, axis=1)[:, np.newaxis]

            # M-step
            for i in xrange(self.nb_states):
                # update priors
                self.priors[i] = np.sum(GAMMA[i, :]) / self.nbData
                self.mu[:, i] = data.T.dot(GAMMA2[i, :].reshape((-1, 1))).T
                mu = np.matlib.repmat(self.mu[:, i].reshape((-1, 1)), 1, self.nbData)
                diff = (data.T - mu)
                self.sigma[i] = diff.dot(np.diag(GAMMA2[i, :])).dot(diff.T) + np.eye(self.nb_dim) * self.reg

            # self.priors = np.mean(GAMMA, axis=1)

            LL[it] = np.sum(np.log(np.sum(L, axis=0))) #/ self.nbData
            # Check for convergence
            if it > nb_min_steps:
                if LL[it] - LL[it - 1] < 0.00001 or it == (maxiter - 1):
                    searching = False

            it += 1

        self.BIC = self.BIC_score(LL[it-1])
        return GAMMA, self.BIC

    def gmr(self, DataIn, in_, out_):

        nbData = np.shape(DataIn)[0]
        nbVarOut = len(out_)
        in_ = in_[0]

        MuTmp = np.zeros((nbVarOut, self.nb_states))
        expData = np.zeros((nbVarOut, nbData))
        expSigma = []
        for i in xrange(nbData):
            expSigma.append(np.zeros((nbVarOut, nbVarOut)))

        H = np.zeros((self.nb_states, nbData))

        for t in xrange(nbData):

            for i in xrange(self.nb_states):
                H[i, t] = self.priors[i] * self.gaussPDF(np.asarray([DataIn[t]]),
                                                         self.mu[in_][i],
                                                         self.sigma[i][in_, in_])

            H[:, t] = H[:, t] / np.sum(H[:, t] + np.finfo(float).tiny)

            for i in xrange(self.nb_states):
                MuTmp[:, i] = self.mu[out_, i] + self.sigma[i][out_, in_] / \
                              self.sigma[i][in_, in_] * \
                              (DataIn[t] - self.mu[in_, i])

                expData[:, t] = expData[:, t] + H[i, t] * MuTmp[:, i]

            for i in xrange(self.nb_states):
                sigma_tmp = self.sigma[i][out_[0]:(out_[-1] + 1), out_[0]:(out_[-1] + 1)] - \
                            self.sigma[i][out_, in_] / self.sigma[i][in_, in_] * self.sigma[i][in_, out_]

                expSigma[t] = expSigma[t] + H[i, t] * (sigma_tmp + MuTmp[:, i].reshape((-1, 1)) * MuTmp[:, i].T)

            expSigma[t] = expSigma[t] - expData[:, t] * expData[:, t].T + np.eye(nbVarOut) * 1E-8

        return expData, expSigma, H

    def gaussPDF(self, x, mean, covar):
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

    def relocateGaussian(self, A, b):
        mu = np.zeros((self._nb_dim, self._nb_states))
        sigma = np.array([np.zeros((self.nb_dim,self.nb_dim)) for i in range(self.nb_states)])

        for i in xrange(self._nb_states):
            temp_mu = np.zeros((self._nb_dim, 1))
            temp_sigma = np.zeros((self.nb_dim, self.nb_dim))
            for frame in xrange(self.frames):
                curr_mu = A[frame].dot(self.mu[:,i].reshape((-1,1))) + b[frame]
                curr_sigma = A[frame] * self.sigma[i] * A[frame].T
                temp_sigma = temp_sigma + np.linalg.inv(curr_sigma)
                temp_mu = temp_mu + np.linalg.inv(curr_sigma).dot(curr_mu)

            sigma[i] = np.linalg.inv(temp_sigma)
            mu[:,i] = sigma[i].dot(temp_mu).flatten().tolist()

        self.sigma = sigma
        self.mu = mu




# p = PatchCollection(patches)
# ax.add_collection(p)
