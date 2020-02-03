#!/usr/bin/env python
# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2020, WPIGaitAnalysisToolKit
#     (www.aimlab.wpi.edu)

#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.

#     \author    <http://www.aimlab.wpi.edu>
#     \author    <nagoldfarb@wpi.edu>
#     \author    Nathaniel Goldfarb
#     \version   0.1
# */
# //==============================================================================


from termcolor import colored


from ..lib.pbdlib.pbdlib import gmm
from ..lib.pbdlib.pbdlib.model import *
from ..lib.pbdlib.pbdlib.functions import gaussPDF
import copy

class GMMWPI(gmm.GMM):

    def __init__(self, nb_states=1, nb_dim=3, init_zeros=False, mu=None, lmbda=None, sigma=None, priors=None):

        # if mu is not None:
        #     nb_states = mu.shape[0]
        #     nb_dim = mu.shape[-1]
        nb_s = nb_states
        nb_d = nb_dim

        gmm.GMM.__init__(self, nb_states=nb_states, nb_dim=nb_dim)
        # flag to indicate that publishing was not init
        self.publish_init = False
        self._mu = mu
        self._lmbda = lmbda
        self._sigma = sigma
        self._priors = priors

        if init_zeros:
            self.init_zeros()

    def kmeansclustering(self, data, reg=1e-8):

        self.reg = reg

        # Criterion to stop the EM iterative update
        cumdist_threshold = 1e-10
        maxIter = 100

        # Initialization of the parameters
        cumdist_old = -1.7977e+308
        nbStep = 0
        self.nbData = data.shape[1]
        idTmp = np.random.permutation(self.nbData)

        Mu = copy.deepcopy(data[:, idTmp[:self.nb_states]])
        searching = True
        distTmpTrans = np.zeros((len(data[0]),self.nb_states, ))
        idList = []

        while searching:

            # E-step %%%%%%%%%%%%%%%%%%%%%%%%%%%%%% %%
            for i in xrange(0, self.nb_states):
                # Compute distances
                thing = np.matlib.repmat(Mu[:, i].reshape((-1, 1)), 1, self.nbData)
                temp = np.power(data - thing, 2.0)
                temp2 = np.sum(temp, 0)
                distTmpTrans[:,i] = temp2

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
            if abs(cumdist - cumdist_old) < cumdist_threshold:
                print 'Maximum number of kmeans iterations, ' + str(maxIter) + ' is reached'
                searching = False

            cumdist_old = cumdist
            nbStep = nbStep + 1

            if nbStep > maxIter:
                print 'Maximum number of kmeans iterations, ' + str(maxIter) + ' is reached'
                searching = False
            print "maxitter ", nbStep

        self.mu = Mu

        return idList


    def init_params_kmeans(self, data):

        idList = self.kmeansclustering(data)
        self.priors = np.ones(self.nb_states) / self.nb_states
        self.sigma = np.array([np.eye(self.nb_dim) for i in range(self.nb_states)])
        self.Trans = np.ones((self.nb_states, self.nb_states)) * 0.01

        for i in xrange(self.nb_states):

            idtmp = np.where(idList==i)
            mat = np.vstack((data[:, idtmp][0][0], data[:, idtmp][1][0]))

            for j in xrange(2, len(data[:,idtmp])):
                mat = np.vstack((mat, data[:, idtmp][j][0]))

            mat = np.concatenate((mat, mat), axis=1)
            self.priors[i] = len(idtmp[0])
            self.sigma[i] = np.cov(mat) + np.eye(self.nb_dim)*self.reg

        self.priors = self.priors / np.sum(self.priors)

    def em(self, data, reg=1e-8, maxiter=100, minstepsize=1e-5, diag=False, reg_finish=False,
           kmeans_init=False, random_init=False, dep_mask=None, verbose=False, only_scikit=False,
           no_init=True):
        """

        :param data:	 		[np.array([nb_timesteps, nb_dim])]
        :param reg:				[list([nb_dim]) or float]
            Regulariazation for EM
        :param maxiter:
        :param minstepsize:
        :param diag:			[bool]
            Use diagonal covariance matrices
        :param reg_finish:		[np.array([nb_dim]) or float]
            Regulariazation for finish step
        :param kmeans_init:		[bool]
            Init components with k-means.
        :param random_init:		[bool]
            Init components randomely.
        :param dep_mask: 		[np.array([nb_dim, nb_dim])]
            Composed of 0 and 1. Mask given the dependencies in the covariance matrices
        :return:
        """

        self.reg = reg

        nb_min_steps = 5  # min num iterations
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
                L [i, :] = self.priors[i] * gaussPDF(data.T, self.mu[:, i], self.sigma[i])

            GAMMA = L / np.sum(L, axis=0)
            GAMMA2 = GAMMA / np.sum(GAMMA, axis=1)[:, np.newaxis]

            # M-step
            for i in xrange(self.nb_states):
                # update priors
                self.priors[i] = np.sum(GAMMA[i,:]) / self.nbData
                self.mu[:, i] = data.T.dot(GAMMA2[i,:].reshape((-1,1))).T
                mu = np.matlib.repmat(self.mu[:, i].reshape((-1, 1)), 1, self.nbData)
                diff = (data.T - mu)
                self.sigma[i] = diff.dot(np.diag(GAMMA2[i,:])).dot(diff.T) + np.eye(self.nb_dim) * self.reg

            # self.priors = np.mean(GAMMA, axis=1)

            LL[it] = np.sum(np.log(np.sum(L, axis=0)))/self.nbData
            # Check for convergence
            if it > nb_min_steps:
                if LL[it] - LL[it - 1] < 1.0000e-04 or it == (100-1):
                    searching = False
            print it
            it += 1
        return GAMMA




    def gmr(self, DataIn,  in_, out_):

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
                H[i, t] = self.priors[i] * gaussPDF(np.asarray([DataIn[t]]),
                                                                    self.mu[in_][i],
                                                                    self.sigma[i][in_, in_])

            H[:, t] = H[:, t] / np.sum(H[:, t] + np.finfo(float).tiny)


            for i in xrange(self.nb_states):
                MuTmp[:, i] = self.mu[out_, i] + self.sigma[i][out_, in_] / \
                              self.sigma[i][in_, in_] * \
                              (DataIn[t] - self.mu[in_, i])

                expData[:, t] = expData[:, t] + H[i, t] * MuTmp[:, i]

            for i in xrange(self.nb_states):
                sigma_tmp = self.sigma[i][out_[0]:(out_[-1]+1), out_[0]:(out_[-1]+1)] - \
                            self.sigma[i][out_, in_] / self.sigma[i][in_, in_] * self.sigma[i][in_, out_]

                expSigma[t] = expSigma[t] + H[i, t] * (sigma_tmp + MuTmp[:, i].reshape((-1,1)) * MuTmp[:, i].T)

            expSigma[t] = expSigma[t] - expData[:, t] * expData[:, t].T + np.eye(nbVarOut) * 1E-8

        return expData, expSigma, H


    def mu(self, value):
        self.nb_dim = value.shape[0]
        self.nb_states = value.shape[1]
        self._mu = value
