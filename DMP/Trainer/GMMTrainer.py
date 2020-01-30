
from Trainer.TrainerBase import TrainerBase
import numpy as np

class GMMTrainer(TrainerBase):

    def __init__(self, data, file_name, n_rf, dt):
        """
           :param file_names: file to save training too
           :param n_rfs: number of DMPs
           :param dt: time step
           :return: None
           """

        super(GMMTrainer, self).__init__(data, file_name, n_rf, dt)

    def writeXML(self):
        pass

    def gen_path(self):
        pass

    def train(self):
        pass

    def getTraj(demos, samples, nbData_):
        nbData = nbData_  # Length of each trajectory
        dt = 0.01
        kp = 50.0
        kv = (2.0 * kp) ** 0.5
        alpha = 1.0
        x_ = None
        dx_ = None
        ddx_ = None
        sIn = []
        taux = []
        tauy = []

        sIn.append(1.0)  # Initialization of decay term
        for t in xrange(1, nbData):
            sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * dt)  # Update of decay term (ds/dt=-alpha s) )

        goal = demos[0][-1]

        for n in xrange(samples):
            demo = demos[n]
            size = demo.shape[0]
            x = utl.spline(np.arange(1, size + 1), demo, np.linspace(1, size, nbData))
            dx = np.divide(np.diff(x, 1), np.power(dt, 1.0))
            dx = np.append([0.0], dx[0])
            ddx = np.divide(np.diff(x, 2), np.power(dt, 2))
            ddx = np.append([0.0, 0.0], ddx[0])
            goals = np.matlib.repmat(goal, nbData, 1)
            tau_ = ddx - (kp * (goals.transpose() - x)) / sIn + (kv * dx) / sIn

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


