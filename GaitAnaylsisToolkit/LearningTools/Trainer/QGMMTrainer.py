import pyquaternion

from . import GMMTrainer as GMMTrainer
from pyquaternion.quaternion import Quaternion as pyq

class QGMMTrainer(GMMTrainer.GMMTrainer):

    def __init__(self, demo, file_name, n_rf, dt=0.01, reg=[1e-5], poly_degree=[15], A=[], b=[]):
        super().__init__(demo, file_name, n_rf, dt)

    def train(self, save=True):
        return super().train(save)

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

            # dx = np.divide(np.diff(x, 1), np.power(self._dt, 1))
            # dx_temp[:x.shape[0],1:x.shape[1]+1] = dx
            dx = self.calc_derivative(x)

            # ddx = np.divide(np.diff(x, 2), np.power(self._dt, 2))
            # ddx_temp[:x.shape[0], 2:x.shape[1] + 2] = ddx
            ddx =  self.calc_derivative(x) #dx_temp

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



    def calc_derivative(self, quat_list, dt=0.01):
        """

        :param quat_list: list of qua
        :param dt:
        :return:
        """

        vel = []
        pyq.Quaternion
        for q in quat_list:
            vel.append(q.derivative(dt))

        return vel

    def resample(self, trajs, poly_degree, resample):
        '''
        This has to be rewritten to
        :param trajs:
        :param poly_degree:
        :param resample:
        :return:
        '''

        t = []
        alpha = 1.0
        demos = []

        # DWT distance function


        idx = np.argmax([l.shape[0] for l in trajs])
        # get the decay term
        t.append(1.0)  #
        for i in range(1, len(trajs[idx])):
            t.append(t[i - 1] - alpha * t[i - 1] * 0.01)  # Update of decay term (ds/dt=-alpha s) )
        t = np.array(t)

        for ii, y in enumerate(trajs):
            dtw_data = {}
            d, cost_matrix, acc_cost_matrix, path = dtw(trajs[idx], y, dist=self.dtw_distance_fnc)
            dtw_data["cost"] = d
            dtw_data["cost_matrix"] = cost_matrix
            dtw_data["acc_cost_matrix"] = acc_cost_matrix
            dtw_data["path"] = path
            data.append(dtw_data)
            data_warp = [y[path[1]]]
            data_warp_rsp = y[path[1]][:x_fit.shape[0]]
            y_fit = y[path[1]][:x_fit.shape[0]]
            temp = [[np.array(ele)] for ele in y_fit.tolist()]
            temp = np.array(temp)
            demos.append(temp)
            dtw_data["unsmooth_path"] = y[path[1]][:x_fit.shape[0]]
            dtw_data["smooth_path"] = temp

        return demos, data

