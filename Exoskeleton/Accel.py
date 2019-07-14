from lib.Exoskeleton.SensorBase import AccelBase


class Accel(AccelBase.AccelBase):

    def __init__(self, name, side, value):
        super(Accel, self).__init__(name, side)
        self.raw_values = value

    @property
    def offset(self):
        return super(Accel, self).offset()

    @property
    def orientation(self):
        return super(Accel, self).orientation()

    def reset(self):
        pass
