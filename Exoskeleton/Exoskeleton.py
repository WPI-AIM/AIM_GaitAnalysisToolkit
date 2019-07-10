import csv
from collections import namedtuple


class Exoskeleton(object):

    def __init__(self, file):
        self._data = {}
        self._leg = namedtuple('leg', 'hip knee ankle fsr')
        self._joint = namedtuple("joint", 'imu, pot')
        self._imu = namedtuple('imu', 'temp accel gyro')
        self._fsr = namedtuple('fsr', 'fsr1 fsr2 fsr3')
        self.joints = ["Hip", "Knee", 'Ankle']
        self._open_exo_file(file)

    def get_data(self):
        return self._data

    def get_left_leg(self):
        return self._left_leg

    def get_right_leg(self):
        return self._right_leg

    def get_center(self):
        return self._center

    def _open_exo_file(self, file_path):
        '''

        :param file_path: path to the data file
        :return: values of the sensors
        :rtype: dict
        '''
        self._data
        with open(file_path, mode='r') as csv_file:

            csv_reader = csv.DictReader(csv_file)
            keys = csv_reader.fieldnames
            print keys
            for key in keys:
                self._data[key] = []
            for row in csv_reader:
                for key in keys:
                    self._data[key].append([float(x.strip()) for x in row[key].split(',')])

        left_fsr = self._fsr(fsr1=self._data['FSR1_Left'], fsr2=self._data['FSR2_Left'],
                             fsr3=self._data['FSR3_Left'])

        right_fsr = self._fsr(fsr1=self._data['FSR1_Right'], fsr2=self._data['FSR2_Right'],
                              fsr3=self._data['FSR3_Right'])

        joints = {}

        for side in ["Right", 'Left']:
            for j in self.joints:
                corrected_name = j
                if j is "Hip":
                    corrected_name = "Knee"
                elif j is "Knee":
                    corrected_name = "Ankle"
                elif j is "Ankle":
                    corrected_name = "Foot"

                joint = "_" + corrected_name

                imu = self._imu(temp=self._data["Temperature_" + side + joint],
                                accel=self._data["Accel_" + side + joint], gyro=self._data["Gyro_" + side + joint])
                joints[side + '_' + j] = self._joint(imu=imu, pot=self._data["Pot_" + side + "_" + j])

        self._right_leg = self._leg(hip=joints["Right_Hip"], knee=joints["Right_Knee"], ankle=joints["Right_Ankle"],
                                    fsr=right_fsr)

        self._left_leg = self._leg(hip=joints["Left_Hip"], knee=joints["Left_Knee"], ankle=joints["Left_Ankle"],
                                   fsr=left_fsr)

        self._center = self._imu(accel=self._data["Accel_Center"], gyro=self._data["Gyro_Center"],
                                 temp=self._data["Temperature_Center"])


if __name__ == '__main__':
    file = "/home/nathaniel/git/exoserver/Main/subject_1234_trial_0.csv"
    data = Exoskeleton(file)
    print data.get_left_leg().ankle.imu.accel
