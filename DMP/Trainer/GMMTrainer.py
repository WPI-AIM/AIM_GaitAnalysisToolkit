
import TrainerBase
import numpy as np
from scipy import signal
from Trajectories import utilities as utl
from Trajectories import GMMWPI
import numpy as np
import math
from lxml import etree
import scipy.interpolate

class GMMTrainer(TrainerBase.TrainerBase):

    def __init__(self, demo, file_name, n_rf, dt):
        """
           :param file_names: file to save training too
           :param n_rfs: number of DMPs
           :param dt: time step
           :return: None
           """
        self._kp = 50.0
        self._kv = (2.0 * self._kp) ** 0.5

        demos = self.resample_demos(demo)
 
        super(GMMTrainer, self).__init__(demos, file_name, n_rf, dt)


    def writeXML(self, expData, expSigma, H):
        """
       Saves the data to a CSV file so that is can be used by a runner
       :param w: weights from training
       :param c: gausian centers
       :param h: Varience of the mean
       :param y0: startpoint
       :param goal: goal
       :return: None
        """

        expData_st = ["%.6f" % number for number in expData]
        expSigma_st = ["%.6f" % number for number in expSigma]
        H_st = ["%.6f" % number for number in H]

        root = etree.Element('RMPs')
        weights = etree.Element('Weights')
        inv_sq_var = etree.Element('inv_sq_var')
        gauss_means = etree.Element('gauss_means')
        y_start = etree.Element('y0')
        the_goal = etree.Element('goal')

        the_goal.text = np.str(np.int(goal))
        y_start.text = np.str(np.int(y0))

        for _w, _c, _h in zip(w_st, c_st, h_st):
            etree.SubElement(weights, "w").text = _w
            etree.SubElement(inv_sq_var, "h").text = _h
            etree.SubElement(gauss_means, "c").text = _c

        root.append(weights)
        root.append(inv_sq_var)
        root.append(gauss_means)
        root.append(the_goal)
        root.append(y_start)
        tree = etree.ElementTree(root)
        tree.write(self._file_name, pretty_print=True, xml_declaration=True, encoding="utf-8")


    def train(self):
        """

        """
        nb_dim = len(self._demo)
        gmm = GMMWPI.GMMWPI(nb_states=self._n_rfs, nb_dim=nb_dim)
        tau, motion, sIn = self.gen_path(self._demo)
        gmm.init_params_kmeans(tau)
        gmm.em(tau, no_init=True)
        expData, expSigma, H = gmm.gmr(sIn, [0], [1])

    @staticmethod
    def resample_demos(self, trajs):
        # find resample length to use
        resample = 100000
        for traj in trajs:
            sample = len(traj)
            resample = min(resample, sample)

        demos = []
        for traj in trajs:
            data = signal.resample(traj, resample)
            temp = []
            for d in data:
                temp.append([np.array(d)])
            temp = np.array(temp)
            demos.append(temp)

        return demos


    def gen_path(self, demos):
        """

        """

        nbData =len(demos[0])
        samples = len(demos)

        alpha = 1.0
        x_ = None
        dx_ = None
        ddx_ = None
        sIn = []
        taux = []

        sIn.append(1.0)  # Initialization of decay term
        for t in xrange(1, nbData):
            sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * dt)  # Update of decay term (ds/dt=-alpha s) )

        goal = demos[0][-1]

        for n in xrange(samples):
            demo = demos[n]
            size = demo.shape[0]
            x = utl.spline(np.arange(1, size + 1), demo, np.linspace(1, size, nbData))
            dx = np.divide(np.diff(x, 1), np.power(self._dt, 1.0))
            dx = np.append([0.0], dx[0])
            ddx = np.divide(np.diff(x, 2), np.power(self._dt, 2))
            ddx = np.append([0.0, 0.0], ddx[0])
            goals = np.matlib.repmat(goal, nbData, 1)
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

        t = sIn * samples
        tau = np.vstack((t, taux))
        motion = np.vstack((x_, dx_, ddx_))

        return tau, motion, sIn


