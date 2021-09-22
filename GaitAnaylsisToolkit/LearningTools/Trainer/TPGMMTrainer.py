from . import TrainerBase
from GaitCore.Core import utilities as utl
from ..Models import TPGMM, GMR
from ..Models.ModelBase import solve_riccati_mat
import numpy as np
from numpy import matlib
from . import GMMTrainer as GMMTrainer


class TPGMMTrainer(GMMTrainer.GMMTrainer):

    def __init__(self, demo, file_name, n_rf, dt=0.01, reg=[1e-5], poly_degree=[15], resample=[False], A=[],b=[]):
        """
       :param file_names: file to save training too
       :param n_rfs: number of DMPs
       :param dt: time step
       :return: None
        """

        if not A and not b:
            try:
                assert len(A) == len(b), "transformations matrix not the same size"
                self.A = A
                self.b = b
            except AssertionError as msg:
                print(msg)
        else:
            self.A = A
            self.b = b

        #
        # rescaled = []
        # self.dtw_data = []
        #
        # if len(reg) == len(demo):
        #     my_reg = [1e-8] + reg
        # else:
        #     my_reg = reg*(1+ len(demo))
        #
        # if len(resample) == len(demo):
        #     my_resample = [1e-8] + reg
        # else:
        #     my_resample = resample*(1+ len(demo))
        #
        # for d, polyD, resamp in zip(demo, poly_degree, my_resample):
        #     demo_, dtw_data_ = self.resample(d, polyD, resamp)
        #     rescaled.append(demo_)
        #     self.dtw_data.append(dtw_data_)

        super(TPGMMTrainer, self).__init__(rescaled, file_name, n_rf, dt, my_reg)

    def train(self, save=True):
        """
        train a model to reproduction
        """
        sIn = []
        alpha = 1.0
        sIn.append(1.0)  # Initialization of decay term

        nb_dim = len(self._demo[0])
        self.gmm = TPGMM.TPGMM(nb_states=self._n_rfs, nb_dim=nb_dim, reg=self.reg)
        taus = []
        goals = []
        for i in range(len(self._demo)):
            tau, motion, goal = self.gen_path(self._demo[i])
            taus.append(tau)
            goals.append(goal)

        # make the decay term
        for t in range(1, self.nbData):
            sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * self._dt)  # Update of decay term (ds/dt=-alpha s) )

        # stack the decay term and all the demos
        t = sIn * self.samples
        tau = np.vstack((t, taus[0]))

        for tau_ in taus[1:]:
            tau = np.vstack((tau, tau_))

        # Make all the transformations
        if not self.A:
            for i in range(len(goals[0])):
                b = [0.0]
                for j in range(len(goals)):
                    b.append(goals[j][i].tolist()[0])
                b = np.asarray(b).reshape((-1, 1))
                self.b.append(b)
                self.A.append(np.eye(len(goals)+1))

        # Do all the transformations
        gammam, BIC = self.gmm.train(tau) # get the goodness of fit
        self.gmm.relocateGaussian(self.A, self.b) # more the gaussian based on the frame transformations
        sigma, mu, priors = self.gmm.get_model() # get all the model parameters
        gmr = GMR.GMR(mu=mu, sigma=sigma, priors=priors) # set up the GMR trainers
        expData, expSigma, H = gmr.train(sIn, [0], range(1, 1+len(self._demo)), self.reg) # train the model
        #ric1 = solve_riccati(expSigma)
        ric2 = solve_riccati_mat(expSigma, self._dt, self.reg) # get the gains for the system

        # save all the data to a dictionary
        self.data["BIC"] = BIC
        self.data["len"] = len(sIn)
        self.data["H"] = H
        self.data["motion"] = motion
        self.data["expData"] = expData
        self.data["expSigma"] = expSigma
        self.data["sIn"] = sIn
        self.data["tau"] = tau
        self.data["mu"] = mu
        self.data["sigma"] = sigma
        self.data["dt"] = self._dt
        self.data["start"] = [self._demo[i][0][0] for i in range(len(self._demo))]
        self.data["goal"] = [self._demo[i][0][-1] for i in range(len(self._demo))]
        self.data["goals"] = goals
        self.data["dtw"] = self.dtw_data
        self.data.update(ric2)

        if save:
            self.save()
        return self.data


    def gen_path(self, demos):
        """

        :param demos: list of training data
        :return:
            - taux: x_hat = x + Kp*dx + Kd*ddx
            - motion: the scaled trajectories
            - ending_pos: last position
        """

        self.nbData = len(demos[0])
        self.samples = len(demos)


        x_ = None
        dx_ = None
        ddx_ = None
        taux = []
        ending_pos = []
        for n in range(self.samples):
            demo = demos[n]
            size = demo.shape[0]

            A = np.eye(demo.shape[1])
            goal = np.array([demos[n][-1]])
            x = utl.spline(np.arange(1, size + 1), demo, np.linspace(1, size, self.nbData))
            sol = np.zeros(x.shape)
            dx_temp = np.zeros(x.shape)
            ddx_temp = np.zeros(x.shape)

            dx = np.divide(np.diff(x, 1), np.power(self._dt, 1))
            dx_temp[:x.shape[0],1:x.shape[1]+1] = dx
            dx = dx_temp

            ddx = np.divide(np.diff(x, 2), np.power(self._dt, 2))
            ddx_temp[:x.shape[0], 2:x.shape[1] + 2] = ddx
            ddx = dx_temp

            goals = np.matlib.repmat(goal, 1, self.nbData)
            x_hat = x + (self._kv/self._kp)*dx + (1.0/self._kp)*ddx
            goals = x_hat - goals

            for i in range(self.nbData):
                sol[:, i] = np.linalg.solve(A, goals[:, i].reshape((-1, 1))).ravel()

            ending_pos.append(demos[n][-1])

            if x_ is not None:
                x_ = x_ + x.tolist()
                dx_ = np.vstack((dx_, dx))
                ddx_ = np.vstack((ddx_, ddx))
            else:
                x_ = x.tolist()
                dx_ = dx.tolist()
                ddx_ = ddx.tolist()

            taux = taux + sol[0].tolist()

        motion = np.vstack((x_, dx_, ddx_))

        return taux, motion, ending_pos



