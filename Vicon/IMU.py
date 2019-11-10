import Devices
from lib.Exoskeleton.Robot import core

class IMU(Devices.Devices):

    def __init__(self, name, sensor):

        self._accel = core.Point(self.sensor["ACCX"],
                                 self.sensor["ACCZ"],
                                 self.sensor["ACCY"])
        self._gyro = core.Point(self.sensor["GYROX"],
                                 self.sensor["GYROZ"],
                                 self.sensor["GYROY"])

        super(IMU, self).__init__(name, sensor, "IMU")


    def get_accel(self):
        """

        :return:
        """
        return self._accel

    def get_gyro(self):
        """

        :return:
        """
        return self._gyro