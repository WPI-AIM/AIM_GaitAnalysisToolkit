from lib.Exoskeleton.SensorBase import FSRBase


class FSR(FSRBase.FSRBase):

    def __init__(self, name, side, value):
        super(FSR, self).__init__(name, side)
        self.raw_values = value

    @property
    def offset(self):
        return super(FSR, self).offset()

    def reset(self):
        pass
