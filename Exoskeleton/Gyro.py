from lib.Exoskeleton.SensorBase import GyroBase, Sensor


class Gyro(GyroBase.GyroBase):

    def __init__(self, name, side, values):
        super(Gyro, self).__init__(name, side)
        self.raw_values = values
        self._type = Sensor.Sensor.GYRO

    @property
    def offset(self):
        return super(Gyro, self).offset()

    @property
    def orientation(self):
        return super(Gyro, self).orientation()

    def reset(self):
        pass
