import math

import numpy as np

# from Sensors import Accel, Gyro, Mag, IMU
from lib.Exoskeleton.SensorBase import IMUBase


# TODO alter to preprocess instead of realtime
class IMU(IMUBase.IMUBase):

    def __init__(self, name, accel, gyro, temp, counter=None, rshal=None):
        """

        :type accel: Accel
        :type gyro: Gyro
        :type mag: Mag

        """
        self._name = name
        self.accel = accel
        self.gyro = gyro
        self.temp = temp
        self.counter = counter
        self.rshal = rshal
        self._orentation = np.array([0, 0, 0])
        self._angular_velocity = np.array([0, 0, 0])
        self._gyro_angle = np.array([0, 0, 0])
        self.kalman = {}
        self.axis = ["x", "y", "z"]
        self.setup_kalman()

    # def setup_kalman(self):
    #     """
    #     set up the kalman filters for each of the axis
    #     :return:
    #     """
    #     dt = 0.001
    #     A = np.matrix([[1, -dt], [0, 1]])
    #     B = np.matrix([[dt], [0]])
    #     C = np.matrix([1, 0])
    #     Cz = np.matrix([1, 0])
    #     Q = np.matrix([[0.1, 0.0], [0.0, 0.1]])
    #     R = 0.1
    #     P = np.matrix([[0.0, 0.0], [0.0, 0.0]])
    #     X = np.matrix([[0], [0]])
    #
    #     self.kalman["x"] = Kalman.Kalman(A, B, C, Q, P, R, X)
    #     self.kalman["y"] = Kalman.Kalman(A, B, C, Q, P, R, X)
    #     self.kalman["z"] = Kalman.Kalman(A, B, Cz, Q, P, R, X)

    @property
    def name(self):
        """

        :return: name of the sensor
        :type str
        """
        return self._name

    def get_accel_angles(self):
        """
        calculate the angles from the accelormeter
        :return: roll and pitch angles
        """
        x, y, z = self.accel.get_values()
        roll = math.atan2(y, z)
        pitch = math.atan2((- x), math.sqrt(y * y + z * z))
        return roll, pitch

    @property
    def orentation(self):
        """
        orentation of the IMU based on the sensor fusion
        :return: orentation
        :type np.array
        """
        return self._orentation

    @orentation.setter
    def orentation(self, orentation):
        self._orentation = orentation

    @property
    def angular_velocity(self):
        return self._angular_velocity

    @angular_velocity.setter
    def angular_velocity(self, angular_velocity):
        self._angular_velocity = angular_velocity

    def update(self):
        """
        updat the angular velcity and orentation of the IMU

        :return: None
        """
        roll, pitch = self.get_accel_angles()
        xdot, ydot, zdot = self.gyro.get_values()
        dt = self.gyro.dt
        state_x = self.update_imu("x", xdot, roll, dt)
        state_y = self.update_imu("y", ydot, pitch, dt)
        state_z = self.update_imu("z", zdot, 0, dt)
        self._orentation = np.array([state_x[0], state_y[0], state_z[0]])
        self._angular_velocity = np.array([state_x[1], state_y[1], state_z[1]])

    def update_imu(self, imu_axis, gyro, accel, dt):
        """
        update an axis of the IMU
        :param imu_axis: str of the axis
        :param gyro: gyro sensor value
        :param accel: accel orentation value
        :param dt: time step
        :return: state of the imu
        """

        A = np.matrix([[1, -dt], [0, 1]])
        B = np.matrix([[dt], [0]])
        self.kalman[imu_axis].A = A
        self.kalman[imu_axis].B = B
        self.kalman[imu_axis].move(gyro, accel)
        state = self.kalman[imu_axis].getState()
        return state
