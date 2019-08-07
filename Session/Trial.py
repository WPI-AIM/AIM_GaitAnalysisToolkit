import math

import numpy as np

from Exoskeleton import Exoskeleton
from Vicon import Vicon
from lib.Exoskeleton.Robot import core


class Trial(object):

    def __init__(self, vicon_file, config_file=None, exo_file=None, dt=None, notes_file=None):

        # self._notes_file = notes_file
        self.names = ["HipAngles", "KneeAngles", "AbsAnkleAngle"]
        self._dt = 100
        self._exoskeleton = Exoskeleton.Exoskeleton(config_file, exo_file)
        self._vicon = Vicon.Vicon(vicon_file)
        self.vicon_set_points = {}
        self._joint_trajs = None
        self._black_list = []
        self.create_index_seperators()

    def create_index_seperators(self):
        """
        This function find the index that seperates
        the sensors by the joints angles
        Sets the varibles:
        self.vicon_set_points
        self.exo_set_points
        :return: None
        """
        offsets = []
        vicon = []
        exo = []
        theta = float(self._exoskeleton.length) / float(self._vicon.length)

        model = self.vicon.get_model_output()
        hip = model.get_right_joint("RHipAngles").angle.x
        N = 10
        hip = np.convolve(hip, np.ones((N,)) / N, mode='valid')

        max_peakind = np.diff(np.sign(np.diff(hip))).flatten()  # the one liner
        max_peakind = np.pad(max_peakind, (1, 10), 'constant', constant_values=(0, 0))
        max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]

        for start in xrange(0, len(max_peakind) - 2):
            error = 10000000
            offset = 0
            starting_value = model.get_left_joint("LHipAngles").angle.x[max_peakind[start]]
            for ii in xrange(0, 25):
                temp_error = model.get_left_joint("LKneeAngles").angle.x[max_peakind[start + 1] + ii]
                if temp_error < error:
                    error = temp_error
                    offset = ii
            offsets.append(offset)

        for ii, start in enumerate(xrange(0, len(max_peakind) - 2)):
            begin = max_peakind[start]
            end = max_peakind[start + 1] + offsets[ii]
            vicon.append((begin, end))
            exo.append((int(math.ceil(theta * begin)), int(math.ceil(theta * end))))

        self.vicon_set_points = vicon  # varible that holds the setpoints for the vicon
        self.exo_set_points = exo  # varible that holds the setpoints for the exo

    def seperate_force_plates(self):
        """
        Seperates then force plate data
        :return: Force plate data
        :rtype: Dict
        """
        joints = {}
        plate1 = self.vicon.get_force_plate(1)
        plate2 = self.vicon.get_force_plate(2)
        plate1_forces = plate1.get_forces()
        plate2_forces = plate2.get_forces()
        plate1_moments = plate1.get_moments()
        plate2_moments = plate2.get_moments()

        p1 = (1, plate1_forces, plate1_moments)
        p2 = (2, plate2_forces, plate2_moments)
        joints[1] = []
        joints[2] = []

        for p in (p1, p2):
            key = p[0]
            plateF = p[1]
            plateM = p[2]
            for inc in self.vicon_set_points:
                start = plate1.get_offset_index(inc[0])
                end = plate1.get_offset_index(inc[1])
                Fx = np.array(plateF.x)[start:end]
                Fy = np.array(plateF.y)[start:end]
                Fz = np.array(plateF.z)[start:end]
                Mx = np.array(plateM.x)[start:end]
                My = np.array(plateM.y)[start:end]
                Mz = np.array(plateM.z)[start:end]
                f = core.Point(Fx, Fy, Fz)
                m = core.Point(Mx, My, Mz)
                data = core.Newton(None, f, m, None)
                time = (len(Fx) / float(self.vicon.length)) * self.dt
                stamp = core.Data(data, np.linspace(0, time, len(data)))
                joints[key].append(stamp)

        return joints

    def seperate_joint_trajectories(self):
        """
        Seperates then joint trajs data
        :return: joint trajectory data
        :rtype: Dict
        """
        joints = {}
        model = self.vicon.get_model_output()
        for fnc, side in zip((model.get_left_joint, model.get_right_joint), ("R", "L")):
            for joint_name in self.names:
                name = side + joint_name
                joints[name] = []
                for inc in self.vicon_set_points:
                    data = np.array(fnc(name).angle.x[inc[0]:inc[1]])
                    time = (len(data) / float(self.vicon.length)) * self.dt
                    stamp = core.Data(data, np.linspace(0, time, len(data)))

                    joints[name].append(stamp)

        return joints

    def seperate_emg(self):
        """
       Seperates then EMGs data
       :return: EMGs data
       :rtype: Core.side
        """
        joints = {}
        emgs = self.vicon.get_all_emgs()

        for key, emg in emgs.iteritems():
            joints[key] = []
            for inc in self.vicon_set_points:
                start = emg.get_offset_index(inc[0])
                end = emg.get_offset_index(inc[1])
                data = np.array(emg.get_values())[start:end]
                time = (len(data) / float(self.vicon.length)) * self.dt
                stamp = core.Data(data, np.linspace(0, time, len(data)))
                joints[key].append(stamp)

        return joints

    def seperate_T_emg(self):
        """
       Seperates then EMGs data
       :return: EMGs data
       :rtype: Core.side
        """
        joints = {}
        emgs = self.vicon.get_all_t_emg()

        for key, emg in emgs.iteritems():
            joints[key] = []
            for inc in self.vicon_set_points:
                start = emg.get_offset_index(inc[0])
                end = emg.get_offset_index(inc[1])
                data = np.array(emg.get_values())[start:end]
                time = (len(data) / float(self.vicon.length)) * self.dt
                stamp = core.Data(data, np.linspace(0, time, len(data)))
                joints[key].append(stamp)

        return joints

    def seperate_CoP(self):
        """
       Seperates then CoP data
       :return: CoP data
       :rtype: Dict
        """

        left = []
        right = []

        left_cop = self.exoskeleton.left_leg.calc_CoP()
        right_cop = self.exoskeleton.right_leg.calc_CoP()

        for inc in self.exo_set_points:
            left_data = left_cop[inc[0]:inc[1]]
            right_data = right_cop[inc[0]:inc[1]]

            time = (len(left_data) / float(self.exoskeleton.length)) * self.dt
            stamp_left = core.Data(left_data, np.linspace(0, time, len(left_data)))
            stamp_right = core.Data(right_data, np.linspace(0, time, len(right_data)))
            left.append(stamp_left)
            right.append(stamp_right)

        side = core.Side(left, right)

        return side

    def seperate_FSR(self):
        """
               Seperates FSR data
               :return: FSR data
               :rtype: Dict
        """

        left_fsr = self.exoskeleton.left_leg.ankle.FSRs
        right_fsr = self.exoskeleton.right_leg.ankle.FSRs
        left = []
        right = []

        for inc in self.exo_set_points:
            left_data = np.array(
                [[left_fsr[0].get_values()[inc[0]:inc[1]]],
                 [left_fsr[1].get_values()[inc[0]:inc[1]]],
                 [left_fsr[2].get_values()[inc[0]:inc[1]]]])

            right_data = np.array(
                [[right_fsr[0].get_values()[inc[0]:inc[1]]],
                 [right_fsr[1].get_values()[inc[0]:inc[1]]],
                 [right_fsr[2].get_values()[inc[0]:inc[1]]]])

            time = (len(left_data) / float(self.exoskeleton.length)) * self.dt
            stamp_left = core.Data(left_data, np.linspace(0, time, len(left_data)))
            stamp_right = core.Data(right_data, np.linspace(0, time, len(right_data)))
            left.append(stamp_left)
            right.append(stamp_right)

        side = core.Side(left, right)

        return side

    def seperate_pots(self):
        """
       Seperates Pot data
       :return: Pot data
       :rtype: Dict
        """
        left_leg = self.exoskeleton.left_leg
        right_leg = self.exoskeleton.right_leg
        left = []
        right = []

        for inc in self.exo_set_points:
            left_data = np.array(
                [[left_leg.hip.pot.get_values()[inc[0]:inc[1]]],
                 [left_leg.knee.pot.get_values()[inc[0]:inc[1]]],
                 [left_leg.ankle.pot.get_values()[inc[0]:inc[1]]]])

            right_data = np.array(
                [[right_leg.hip.pot.get_values()[inc[0]:inc[1]]],
                 [right_leg.knee.pot.get_values()[inc[0]:inc[1]]],
                 [right_leg.ankle.pot.get_values()[inc[0]:inc[1]]]])

            time = (len(left_data) / float(self.exoskeleton.length)) * self.dt

            stamp_left = core.Data()
            stamp_right = core.Data()
            stamp_right.data = right_data
            stamp_left.data = left_data
            stamp_left.time = np.linspace(0, time, len(left_data))
            stamp_right.time = np.linspace(0, time, len(right_data))
            left.append(stamp_left)
            right.append(stamp_right)

        side = core.Side(left, right)
        return side

    def seperate_accel(self):
        """
                Seperates then force plate data
                :return: Force plate data
                :rtype: Dict
                """
        left_leg = self.exoskeleton.left_leg
        right_leg = self.exoskeleton.right_leg

        left = []
        right = []

        for inc in self.exo_set_points:
            left_data = np.array(
                [[left_leg.hip.IMU.accel.get_values()[inc[0]:inc[1]]],
                 [left_leg.knee.IMU.accel.get_values()[inc[0]:inc[1]]],
                 [left_leg.ankle.IMU.accel.get_values()[inc[0]:inc[1]]]])

            right_data = np.array(
                [[right_leg.hip.IMU.accel.get_values()[inc[0]:inc[1]]],
                 [right_leg.knee.IMU.accel.get_values()[inc[0]:inc[1]]],
                 [right_leg.ankle.IMU.accel.get_values()[inc[0]:inc[1]]]])

            time = (len(left_data) / float(self.exoskeleton.length)) * self.dt

            stamp_left = core.Data(left_data, np.linspace(0, time, len(left_data)))
            stamp_right = core.Data(right_data, np.linspace(0, time, len(right_data)))
            left.append(stamp_left)
            right.append(stamp_right)

        side = core.Side(left, right)

        return side

    def seperate_gyro(self):
        """
                Seperates then force plate data
                :return: Force plate data
                :rtype: Dict
                """
        left_leg = self.exoskeleton.left_leg
        right_leg = self.exoskeleton.right_leg

        left = []
        right = []

        for inc in self.exo_set_points:
            left_data = np.array(
                [[left_leg.hip.IMU.gyro.get_values()[inc[0]:inc[1]]],
                 [left_leg.knee.IMU.gyro.get_values()[inc[0]:inc[1]]],
                 [left_leg.ankle.IMU.gyro.get_values()[inc[0]:inc[1]]]])

            right_data = np.array(
                [[right_leg.hip.IMU.gyro.get_values()[inc[0]:inc[1]]],
                 [right_leg.knee.IMU.gyro.get_values()[inc[0]:inc[1]]],
                 [right_leg.ankle.IMU.gyro.get_values()[inc[0]:inc[1]]]])

            time = (len(left_data) / float(self.exoskeleton.length)) * self.dt
            stamp_left = core.Data(left_data, np.linspace(0, time, len(left_data)))
            stamp_right = core.Data(right_data, np.linspace(0, time, len(right_data)))
            left.append(stamp_left)
            right.append(stamp_right)

        side = core.Side(left, right)
        return side

    @property
    def dt(self):
        return self._dt

    @property
    def exoskeleton(self):
        return self._exoskeleton

    @property
    def vicon(self):
        return self._vicon

    @property
    def joint_trajs(self):
        return self._joint_trajs

    @dt.setter
    def dt(self, value):
        self._dt = value

    @exoskeleton.setter
    def exoskeleton(self, value):
        self._exoskeleton = value

    @vicon.setter
    def vicon(self, value):
        self._vicon = value

    @joint_trajs.setter
    def joint_trajs(self, value):
        self._joint_trajs = value


if __name__ == '__main__':
    vicon_file = "/media/nathaniel/Data/LowerLimb_HealthyGait/Subject02/walking/Walking01.csv"
    config_file = "/home/nathaniel/git/exoserver/Config/sensor_list.yaml"
    exo_file = "/home/nathaniel/git/exoserver/Main/subject_1234_trial_1.csv"
    trial = Trial(vicon_file, config_file, exo_file)
    joints = trial.seperate_joint_trajectories()
    plate = trial.seperate_force_plates()
    left, right = trial.seperate_CoP()
    emg = trial.seperate_emg()
