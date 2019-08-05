from lib.Exoskeleton.SensorBase import AccelBase


class Accel(AccelBase.AccelBase):

    def __init__(self, name, side, value):
        """
        Derived class to hold data

        :param name: name of the sensors
        :param side: side of the sensor
        :param value: array of the values
        :type name: str
        :type side: str
        :type value: list
        """
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
