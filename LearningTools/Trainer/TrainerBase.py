
import abc
import numpy as np
import numpy.polynomial.polynomial as poly
from dtw import dtw


class TrainerBase(object):

    def __init__(self, demo, file_name, n_rfs, dt):
        """
           :param file_names: file to save training too
           :param n_rfs: number of DMPs
           :param dt: time step
           :return: None
           """
        self._demo = demo
        self._file_name = file_name
        self._n_rfs = n_rfs
        self._dt = dt
        self.data = {}


    def resample(self, trajs):


        manhattan_distance = lambda x, y: np.abs(x - y)
        ecuild_distance = lambda x, y: np.sqrt(x*x + y*y)

        idx = np.argmax([l.shape[0] for l in trajs])
        t = []
        alpha = 1.0
        t.append(1.0)  # Initialization of decay term
        for i in xrange(1, len(trajs[idx])):
            t.append(t[i - 1] - alpha * t[i - 1] * 0.01)  # Update of decay term (ds/dt=-alpha s) )
        t = np.array(t)

        demos = []
        coefs = poly.polyfit(t, trajs[idx], 20)
        ffit = poly.Polynomial(coefs)  # instead of np.poly1d
        x_fit =  ffit(t)
        data = []

        for ii, y in enumerate(trajs):
            dtw_data = {}
            d, cost_matrix, acc_cost_matrix, path = dtw(x_fit, y, dist=manhattan_distance)
            dtw_data["cost"] = d
            dtw_data["cost_matrix"] = cost_matrix
            dtw_data["acc_cost_matrix"] = acc_cost_matrix
            dtw_data["path"] = path
            data.append(dtw_data)
            data_warp = [y[path[1]][:x_fit.shape[0]]]
            coefs = poly.polyfit(t, data_warp[0], 20)
            ffit = poly.Polynomial(coefs)  # instead of np.poly1d
            y_fit = ffit(t)
            temp = [[np.array(ele)] for ele in y_fit.tolist()]
            temp = np.array(temp)
            demos.append(temp)
        return demos, data

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def gen_path(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass


def calculate_imitation_metric_spatially(demos, imitation):
    M = len(demos)
    T = len(imitation)
    metric = 0.0

    for m in xrange(M):
        for t in xrange(T):
            metric += np.sqrt(np.power(demos[m][t] - imitation[t]))

    return metric


