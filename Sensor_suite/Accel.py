from Sensor_suite import PostSensor


class Accel(PostSensor.PostSensor):

    def __init__(self, name, sensor):
        super(Accel, self).__init__(name, sensor, "Accel")

    def get_x(self):
        return self.sensor["ACCX"]

    def get_y(self):
        return self.sensor["ACCY"]

    def get_z(self):
        return self.sensor["ACCZ"]
