
import abc

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

    @abc.abstractmethod
    def save(self):
        pass

    @abc.abstractmethod
    def gen_path(self):
        pass

    @abc.abstractmethod
    def train(self):
        pass

