import Devices


class ForcePlate(Devices.Devices):

    def __init__(self, name, forces, moments):
        sensor = {"force": forces, "moment": moments}
        super(ForcePlate, self).__init__(name, sensor, "IMU")

    def forces(self):
        return self.sensor["forces"]

    def moments(self):
        return self.sensor["moments"]

    def get_moment_x(self):
        return self.moments["Mx"]["data"]

    def get_moment_y(self):
        return self.moments["My"]["data"]

    def get_moment_z(self):
        return self.moments["Mz"]["data"]

    def get_force_x(self):
        return self.forces["Fx"]["data"]

    def get_force_y(self):
        return self.forces["Fy"]["data"]

    def get_foce_z(self):
        return self.forces["Fz"]["data"]
