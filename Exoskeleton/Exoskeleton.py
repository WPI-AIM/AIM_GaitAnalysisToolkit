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
        self._open_exo_file(config_path, data_path)
        self._setup_robot()

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
                print keys
                for key in keys:
                    self._data[key] = []
                for row in csv_reader:
                    for key in keys:
                        self._data[key].append([float(x.strip()) for x in row[key].split(',')])

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
                    self.sensors[name] = FSR.FSR(name, side, self._data[name])
                    print orientation
                    self.sensors[name].orientation = orientation
                elif sensor_type == "Pot":
                    self.sensors[name] = Pot.Pot(name, side, self._data[name])

            # set up IMUs
            for name in config:
                item = config[name]
                if name == "IMU":
                    accel = item.get("accel")
                    gyro = item.get("gyro")
                    temp = item.get("temp")
                    counter = item.get("counter")
                    rshal = item.get("rshal")
                    imu = IMU.IMU(name,
                                  self.sensors[accel],
                                  self.sensors[gyro],
                                  self.sensors[temp])

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
        self.right_leg = Leg.Leg(right_hip, right_knee, right_ankle)
        left_hip = Joint.Joint(self._imus["IMU_Left_Hip"], self.sensors["Pot_Left_Hip"])
        left_knee = Joint.Joint(self._imus["IMU_Left_Knee"], self.sensors["Pot_Left_Knee"])
        fsr = [self.sensors["FSR1_Left"], self.sensors["FSR2_Left"], self.sensors["FSR3_Left"]]
        left_ankle = Joint.Joint(self._imus["IMU_Left_Ankle"], self.sensors["Pot_Left_Ankle"], fsr)
        self.left_leg = Leg.Leg(left_hip, left_knee, left_ankle)

    # def _open_exo_file(self, file_path):
    #     '''
    #
    #     :param file_path: path to the data file
    #     :return: values of the sensors
    #     :rtype: dict
    #     '''
    #     self._data
    #     with open(file_path, mode='r') as csv_file:
    #
    #         csv_reader = csv.DictReader(csv_file)
    #         keys = csv_reader.fieldnames
    #         print keys
    #         for key in keys:
    #             self._data[key] = []
    #         for row in csv_reader:
    #             for key in keys:
    #                 self._data[key].append([float(x.strip()) for x in row[key].split(',')])
    #
    #     left_fsr = self._fsr(fsr1=self._data['FSR1_Left'], fsr2=self._data['FSR2_Left'],
    #                          fsr3=self._data['FSR3_Left'])
    #
    #     right_fsr = self._fsr(fsr1=self._data['FSR1_Right'], fsr2=self._data['FSR2_Right'],
    #                           fsr3=self._data['FSR3_Right'])
    #
    #     joints = {}
    #
    #     for side in ["Right", 'Left']:
    #         for j in self.joints:
    #             corrected_name = j
    #             if j is "Hip":
    #                 corrected_name = "Knee"
    #             elif j is "Knee":
    #                 corrected_name = "Ankle"
    #             elif j is "Ankle":
    #                 corrected_name = "Foot"
    #
    #             joint = "_" + corrected_name
    #
    #             imu = self._imu(temp=self._data["Temperature_" + side + joint],
    #                             accel=self._data["Accel_" + side + joint], gyro=self._data["Gyro_" + side + joint])
    #             joints[side + '_' + j] = self._joint(imu=imu, pot=self._data["Pot_" + side + "_" + j])
    #
    #     self._right_leg = self._leg(hip=joints["Right_Hip"], knee=joints["Right_Knee"], ankle=joints["Right_Ankle"],
    #                                 fsr=right_fsr)
    #
    #     self._left_leg = self._leg(hip=joints["Left_Hip"], knee=joints["Left_Knee"], ankle=joints["Left_Ankle"],
    #                                fsr=left_fsr)
    #
    #     self._center = self._imu(accel=self._data["Accel_Center"], gyro=self._data["Gyro_Center"],
    #                              temp=self._data["Temperature_Center"])


if __name__ == '__main__':
    file = "/home/nathaniel/git/exoserver/Main/subject_1234_trial_0.csv"
    data = Exoskeleton(file)
