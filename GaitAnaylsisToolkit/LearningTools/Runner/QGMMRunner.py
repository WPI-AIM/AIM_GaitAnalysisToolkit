
from . import TPGMMRunner as TPGMMRunner


class QGMMRunner(TPGMMRunner.TPGMMRunner):

    def step(self, x=None, dx=None):
        return super().step(x, dx)