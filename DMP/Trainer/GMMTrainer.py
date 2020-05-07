
import TrainerBase
import numpy as np
from scipy import signal
from lib.GaitAnalysisToolkit.lib.GaitCore.Core import utilities as utl
from ...Trajectories import GMMWPI
import numpy as np
import numpy.polynomial.polynomial as poly
import numpy.matlib
from dtw import dtw
import math
from lxml import etree
import scipy.interpolate
import pickle

class GMMTrainer(TrainerBase.TrainerBase):

    def __init__(self, demo, file_name, n_rf, dt=0.01):
        """
           :param file_names: file to save training too
           :param n_rfs: number of DMPs
           :param dt: time step
           :return: None
           """
        self._kp = 50.0
        self._kv = (2.0 * self._kp) ** 0.5
        demos2, self.dtw_data = self.resample(demo)
        super(GMMTrainer, self).__init__(demos2, file_name, n_rf, dt)


    def save(self, expData, expSigma, H, sIn, tau, motion):
        """
       Saves the data to a CSV file so that is can be used by a runner
       :param w: weights from training
       :param c: gausian centers
       :param h: Varience of the mean
       :param y0: startpoint
       :param goal: goal
       :return: None
        """

        data = {}

        data["len"] = len(sIn)
        data["H"] = H
        data["motion"] = motion
        data["expData"] = expData
        data["expSigma"] = expSigma
        data["sIn"] = sIn
        data["mu"] = self.gmm.mu
        data["tau"] = tau
        data["sigma"] = self.gmm.sigma
        data["dt"] = self._dt
        data["start"] = self._demo[0][0]
        data["goal"] = self._demo[0][-1]
        data["dtw"] = self.dtw_data
        data["BIC"] = self.BIC
        with open(self._file_name + '.pickle', 'wb') as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def train(self, save=True):
        """

        """
        nb_dim = len(self._demo)
        self.gmm = GMMWPI.GMMWPI(nb_states=self._n_rfs, nb_dim=nb_dim)
        tau, motion, sIn = self.gen_path(self._demo)
        self.gmm.init_params_kmeans(tau)
        gammam, self.BIC = self.gmm.em(tau)
        expData, expSigma, H = self.gmm.gmr(sIn, [0], [1])
        if save:
            self.save(expData, expSigma, H, sIn, tau, motion)
        return self.BIC


    def gen_path(self, demos):
        """

        """

        self.nbData = len(demos[0])
        self.samples = len(demos)

        alpha = 1.0
        x_ = None
        dx_ = None
        ddx_ = None
        sIn = []
        taux = []

        sIn.append(1.0)  # Initialization of decay term
        for t in xrange(1, self.nbData):
            sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * self._dt)  # Update of decay term (ds/dt=-alpha s) )

        goal = demos[0][-1]

        for n in xrange(self.samples):
            demo = demos[n]
            size = demo.shape[0]
            x = utl.spline(np.arange(1, size + 1), demo, np.linspace(1, size, self.nbData))
            dx = np.divide(np.diff(x, 1), np.power(self._dt, 1))
            dx = np.append([0.0], dx[0])
            ddx = np.divide(np.diff(x, 2), np.power(self._dt, 2))
            ddx = np.append([0.0, 0.0], ddx[0])
            goals = np.matlib.repmat(goal, self.nbData, 1)
            tau_ = ddx - (self._kp * (goals.transpose() - x)) / sIn + (self._kv * dx) / sIn

            if x_ is not None:
                x_ = x_ + x.tolist()
                dx_ = np.vstack((dx_, dx))
                ddx_ = np.vstack((ddx_, ddx))
            else:
                x_ = x.tolist()
                dx_ = dx.tolist()
                ddx_ = ddx.tolist()

            taux = taux + tau_[0].tolist()

        t = sIn * self.samples
        tau = np.vstack((t, taux))
        motion = np.vstack((x_, dx_, ddx_))

        return tau, motion, sIn


