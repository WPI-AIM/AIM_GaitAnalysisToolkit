import Devices
from lib.Exoskeleton.Robot import core

class Accel(Devices.Devices):

    def __init__(self, name, sensor):
        self._accel = core.Point(self.sensor["ACCX"],
                                self.sensor["ACCZ"],
                                self.sensor["ACCY"])
        super(Accel, self).__init__(name, sensor, "Accel")

    def get(self):
        return self._accel
