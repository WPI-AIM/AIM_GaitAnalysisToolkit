import csv

import yaml

import Accel
import FSR
import Gyro
import IMU
import Leg
import Pot
from lib.Exoskeleton.Robot import ExoskeletonBase, Joint


class Exoskeleton(ExoskeletonBase.ExoskeletonBase):

    def __init__(self, config_path, data_path):
        self._data = {}
        self.sensors = {}
        self._imus = {}
        # self._open_exo_file(config_path, data_path)
        self._setup_sensors(config_path, data_path)
        self._setup_robot()
        self._length = len(self.right_leg._ankle.pot.get_values())

    @property
    def length(self):
        return self._length

    @length.setter
    def length(self, value):
        self._length = value

    def _setup_sensors(self, config_path, data_path):
        """
        This sets up the sensors and adds it up to the SM
        :param config_path: config yaml file
        :type config_path: str
        :return:
        """

        # open and load the yaml file
        with open(config_path, "r") as loader:
            config = yaml.load(loader)

            with open(data_path, mode='r') as csv_file:

                csv_reader = csv.DictReader(csv_file)
                keys = csv_reader.fieldnames
                for key in keys:
                    self._data[key] = []
                for row in csv_reader:
                    for key in keys:
                        self._data[key].append([float(x.strip()) for x in row[key].split(',')])
            self._length = len(self._data["FSR2_Right"])
            for name in config:

                item = config[name]
                byte_list = [item.get("block1"), item.get("block2")]
                sensor_type = item.get("type")
                location = item.get("location")
                side = item.get("side")
                axis = item.get("axis")
                orientation = item.get("orientation")

                if sensor_type == "Accel":
                    self.sensors[name] = Accel.Accel(name, side, self._data[name])
                elif sensor_type == "Gyro":
                    self.sensors[name] = Gyro.Gyro(name, side, self._data[name])
                elif sensor_type == "FSR":
                    data = [item for sublist in self._data[name] for item in sublist]
                    self.sensors[name] = FSR.FSR(name, side, data)
                    self.sensors[name].orientation = orientation
                elif sensor_type == "Pot":
                    data = [item for sublist in self._data[name] for item in sublist]
                    self.sensors[name] = Pot.Pot(name, side, data)

            # set up IMUs
            for name in config:
                item = config[name]
                if "IMU" in name:
                    accel = item.get("accel")
                    gyro = item.get("gyro")
                    temp = item.get("temp")
                    counter = item.get("counter")
                    rshal = item.get("rshal")
                    imu = IMU.IMU(name,
                                  self.sensors[accel],
                                  self.sensors[gyro])

                    self._imus[name] = imu

    def _setup_robot(self):
        """
        create all the joints and link in the robot
        :return:
        """
        right_hip = Joint.Joint(self._imus["IMU_Right_Hip"], self.sensors["Pot_Right_Hip"])
        right_knee = Joint.Joint(self._imus["IMU_Right_Knee"], self.sensors["Pot_Right_Knee"])
        fsr = [self.sensors["FSR1_Right"], self.sensors["FSR2_Right"], self.sensors["FSR3_Right"]]
        right_ankle = Joint.Joint(self._imus["IMU_Right_Ankle"], self.sensors["Pot_Right_Ankle"], fsr)
        self._right_leg = Leg.Leg(right_hip, right_knee, right_ankle)
        left_hip = Joint.Joint(self._imus["IMU_Left_Hip"], self.sensors["Pot_Left_Hip"])
        left_knee = Joint.Joint(self._imus["IMU_Left_Knee"], self.sensors["Pot_Left_Knee"])
        fsr = [self.sensors["FSR1_Left"], self.sensors["FSR2_Left"], self.sensors["FSR3_Left"]]
        left_ankle = Joint.Joint(self._imus["IMU_Left_Ankle"], self.sensors["Pot_Left_Ankle"], fsr)
        self._left_leg = Leg.Leg(left_hip, left_knee, left_ankle)

    @property
    def left_leg(self):
        """
         :rtype Leg.Leg
         """
        return self._left_leg

    @left_leg.setter
    def left_leg(self, value):

        self._left_leg = value

    @property
    def right_leg(self):
        """
         :rtype Leg.Leg
         """
        return self._right_leg

    @right_leg.setter
    def right_leg(self, value):
        self._right_leg = value

if __name__ == '__main__':
    file = "/home/nathaniel/git/exoserver/Main/subject_1234_trial_0.csv"
    data = Exoskeleton(file)
