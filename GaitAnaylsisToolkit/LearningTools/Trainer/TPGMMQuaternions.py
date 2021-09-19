
import TPGMMTrainer


class TPGMMQuaternions(TPGMMTrainer.TPGMMTrainer):

    def __init__(self, demo, file_name, n_rf, dt=0.01, reg=[1e-5], poly_degree=[15], A=[], b=[]):
        super().__init__(demo, file_name, n_rf, dt, reg, poly_degree, A, b)

    def train(self, save=True):
        return super().train(save)