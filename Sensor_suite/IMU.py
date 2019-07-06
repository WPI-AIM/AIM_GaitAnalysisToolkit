from Sensor_suite import PostSensor


class IMU(PostSensor):

    def __init__(self, name, sensor):
        super(IMU, self).__init__(name, sensor, "IMU")

    def get_accel_x(self):
        return self.sensor["ACCX"]

    def get_accel_y(self):
        return self.sensor["ACCY"]

    def get_accel_z(self):
        return self.sensor["ACCZ"]

    def get_gyro_x(self):
        return self.sensor["GYROX"]

    def get_gyro_y(self):
        return self.sensor["GYROY"]

    def get_gyro_z(self):
        return self.sensor["GYROZ"]

    def get_mag_x(self):
        return self.sensor["MAGX"]

    def get_mag_y(self):
        return self.sensor["MAGY"]

    def get_mag_z(self):
        return self.sensor["MAGZ"]
