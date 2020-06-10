from . import TrainerBase


class DMPTrainer(TrainerBase):

    def __init__(self, file_name, n_rf, dt):
        """
           :param file_names: file to save training too
           :param n_rfs: number of DMPs
           :param dt: time step
           :return: None
           """

        super(DMPTrainer, self).__init__(file_name, n_rf, dt)
        
    def writeXML(self):
        pass

    def gen_path(self):
        pass

    def train(self):
        pass


