
import TrainerBase
from lib.GaitAnalysisToolkit.lib.GaitCore.Core import utilities as utl
from lib.GaitAnalysisToolkit.LearningTools.Models import TPGMM, GMR
from lib.GaitAnalysisToolkit.LearningTools.Models.ModelBase import solve_riccati_mat, solve_riccati
import numpy as np
import numpy.matlib


class TPGMMTrainer(TrainerBase.TrainerBase):

    def __init__(self, demo, file_name, n_rf, dt=0.01, poly_degree=15):
        """
           :param file_names: file to save training too
           :param n_rfs: number of DMPs
           :param dt: time step
           :return: None
           """
        self._kp = 50.0
        self._kv = (2.0 * self._kp) ** 0.5
        self.A = []
        self.b = []
        rescaled = []
        self.dtw_data = []
        for d in demo:
            demo_, dtw_data_ = self.resample(d, poly_degree)
            rescaled.append(demo_)
            self.dtw_data.append(dtw_data_)

        super(TPGMMTrainer, self).__init__(rescaled, file_name, n_rf, dt)


    def train(self, save=True):
        """
        train a model to reproduction
        """
        sIn = []
        alpha = 1.0
        sIn.append(1.0)  # Initialization of decay term

        nb_dim = len(self._demo[0])
        self.gmm = TPGMM.TPGMM(nb_states=self._n_rfs, nb_dim=nb_dim)
        taus = []
        goals = []
        for i in xrange(len(self._demo)):
            tau, motion, goal = self.gen_path(self._demo[i])
            taus.append(tau)
            goals.append(goal)

        for t in xrange(1, self.nbData):
            sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * self._dt)  # Update of decay term (ds/dt=-alpha s) )

        t = sIn * self.samples
        tau = np.vstack((t, taus[0]))

        for tau_ in taus[1:]:
            tau = np.vstack((tau, tau_))

        for i in xrange(len(goals[0])):
            b = [0.0]
            for j in xrange(len(goals)):
                b.append(goals[j][i].tolist()[0])
            b = np.asarray(b).reshape((-1, 1))
            self.b.append(b)
            self.A.append(np.eye(len(goals)+1))

        gammam, BIC = self.gmm.train(tau)
        self.gmm.relocateGaussian(self.A, self.b)
        sigma, mu, priors = self.gmm.get_model()
        gmr = GMR.GMR(mu=mu, sigma=sigma, priors=priors)
        expData, expSigma, H = gmr.train(sIn, [0], range(1, 1+len(self._demo)))
        #ric1 = solve_riccati(expSigma)
        ric2 = solve_riccati_mat(expSigma)

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
        self.data["start"] = [self._demo[i][0][0] for i in xrange(len(self._demo))]
        self.data["goal"] = [self._demo[i][0][-1] for i in xrange(len(self._demo))]
        self.data["dtw"] = self.dtw_data
        self.data.update(ric2)

        if save:
            self.save()
        return self.data


    def gen_path(self, demos):
        """

        """

        self.nbData = len(demos[0])
        self.samples = len(demos)


        x_ = None
        dx_ = None
        ddx_ = None
        sIn = []
        taux = []
        ending_pos = []
        for n in xrange(self.samples):
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

            for i in xrange(self.nbData):
                sol[:, i] = np.linalg.solve(A, goals[:, i].reshape((-1, 1))).ravel()

            # self.A.append(np.eye(2))
            # self.b.append(np.array([[0.0], [demos[n][-1]]]))
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



