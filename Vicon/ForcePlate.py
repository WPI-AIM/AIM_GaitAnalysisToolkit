from collections import namedtuple

import Devices


class ForcePlate(Devices.Devices):

    def __init__(self, name, forces, moments):
        self._Point = namedtuple('Point', 'x y z')
        self._Newton = namedtuple('Newton', 'angle force moment power')
        self.force = self._Point(forces["Fx"]["data"], forces["Fy"]["data"], forces["Fz"]["data"])
        self.moment = self._Point(moments["Mx"]["data"], moments["My"]["data"], moments["Mz"]["data"])
        sensor = self._Newton(None, self.force, self.moment, None)
        super(ForcePlate, self).__init__(name, sensor, "IMU")

    def get_forces(self):
        return self._sensor.force

    def get_moments(self):
        return self._sensor.moment
