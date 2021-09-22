
from . import GMMTrainer as GMMTrainer


class QGMMTrainer(GMMTrainer.GMMTrainer):

    def __init__(self, demo, file_name, n_rf, dt=0.01, reg=[1e-5], poly_degree=[15], A=[], b=[]):
        super().__init__(demo, file_name, n_rf, dt)

    def train(self, save=True):
        return super().train(save)

    def gen_path(self, demos):
        '''
        This needs to be rewritten
        :param demos:
        :return:
        '''
        return super().gen_path(demos)

    def resample(self, trajs, poly_degree, resample):
        '''
        This has to be rewritten to
        :param trajs:
        :param poly_degree:
        :param resample:
        :return:
        '''
        return super().resample(trajs, poly_degree, resample)