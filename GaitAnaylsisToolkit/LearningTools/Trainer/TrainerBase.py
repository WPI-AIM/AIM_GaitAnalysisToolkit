
import abc
import numpy as np
import numpy.polynomial.polynomial as poly
from dtw import dtw
import pickle
from scipy import signal
import matplotlib.pyplot as plt
class TrainerBase(object):

    def __init__(self, demo, file_name, n_rfs=15, dt=0.01, reg=1e-8):
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
        self._reg = reg

    @property
    def reg(self):
        return self._reg

    @reg.setter
    def reg(self, value):
        self._reg = value

    def resample(self, trajs, poly_degree, resample):
        """

        :param trajs: list of demos
        :param poly_degree: degree of polynomal to smooth
        :return:
        """

        t = []
        alpha = 1.0
        demos = []

        # DWT distance function
        manhattan_distance = lambda x, y: np.abs(x - y)

        idx = np.argmax([l.shape[0] for l in trajs])
        # get the decay term
        t.append(1.0)  #
        for i in range(1, len(trajs[idx])):
            t.append(t[i - 1] - alpha * t[i - 1] * 0.01)  # Update of decay term (ds/dt=-alpha s) )
        t = np.array(t)

        # Smooth the data
        coefs = poly.polyfit(t, trajs[idx], 20)
        ffit = poly.Polynomial(coefs)  # instead of np.poly1d
        x_fit = ffit(t)
        data = []

        # scale all the data using DTW
        for ii, y in enumerate(trajs):
            dtw_data = {}
            d, cost_matrix, acc_cost_matrix, path = dtw(trajs[idx], y, dist=manhattan_distance)
            dtw_data["cost"] = d
            dtw_data["cost_matrix"] = cost_matrix
            dtw_data["acc_cost_matrix"] = acc_cost_matrix
            dtw_data["path"] = path
            data.append(dtw_data)
            data_warp = [y[path[1]]]
            data_warp_rsp = y[path[1]][:x_fit.shape[0]]
            if resample:
                data_warp_rsp = signal.resample(data_warp[0], x_fit.shape[0])  # resample dtw output 188 points to 118 points
            coefs = poly.polyfit(t, data_warp_rsp, poly_degree)
            ffit = poly.Polynomial(coefs)  # instead of np.poly1d
            y_fit = ffit(t)
            temp = [[np.array(ele)] for ele in y_fit.tolist()]
            temp = np.array(temp)
            demos.append(temp)
            dtw_data["unsmooth_path"] =y[path[1]][:x_fit.shape[0]]
            dtw_data["smooth_path"] = temp
        return demos, data

    @abc.abstractmethod
    def save(self):
        """
       Saves the data to a CSV file so that is can be used by a runner
       :return: None
        """
        with open(self._file_name + '.pickle', 'wb') as handle:
            pickle.dump(self.data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    @abc.abstractmethod
    def gen_path(self):
        pass

    @abc.abstractmethod
    def train(self, save="True"):
        """

        :param save: Boolean to save the model
        :return:
        """
        pass


def calculate_imitation_metric_spatially(demos, imitation):
    M = len(demos)
    T = len(imitation)
    metric = 0.0

    for m in range(M):
        for t in range(T):
            metric += np.sqrt(np.power(demos[m][t] - imitation[t]))

    return metric


