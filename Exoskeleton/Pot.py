# import Sensor
from lib.Exoskeleton.SensorBase import PotBase


class Pot(PotBase.PotBase):

    def __init__(self, name, side, values):
        """
               Derived class to hold data

               :param name: name of the sensors
               :param side: side of the sensor
               :param value: array of the values
               :type name: str
               :type side: str
               :type value: list
        """
        super(Pot, self).__init__(name, side)
        self.raw_values = values

    @property
    def offset(self):
        return super(Pot, self).offset()

    @property
    def orientation(self):
        return super(Pot, self).orientation()

    def reset(self):
        pass

    def get_angle(self):
        return self.filtered_values - self.offset
