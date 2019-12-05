import math

import numpy as np
from Exoskeleton import Exoskeleton
from Vicon import Vicon
from lib.Exoskeleton.Robot import core
import lib.Plotting_Tools as PT
import math


class Trial(object):

    def __init__(self, vicon_file, config_file=None, exo_file=None, dt=.01, notes_file=None):

        # self._notes_file = notes_file
        self.names = ["HipAngles", "KneeAngles", "AbsAnkleAngle"]
        self._dt = dt
        if config_file is not None and exo_file is not None:
            self._using_exo = True
            self._exoskeleton = Exoskeleton.Exoskeleton(config_file, exo_file)
        else:
            self._using_exo = False
        self._vicon = Vicon.Vicon(vicon_file)
        self.vicon_set_points = {}
        self._joint_trajs = None
        self._black_list = []
        self._use_black_list = False
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
        if self._using_exo:
            theta = float(self._exoskeleton.length) / float(self._vicon.length)
        else:
            theta = 0

        model = self.vicon.get_model_output()
        hip = model.get_right_leg().hip.angle.x
        N = 10
        hip = np.convolve(hip, np.ones((N,)) / N, mode='valid')

        max_peakind = np.diff(np.sign(np.diff(hip))).flatten()  # the one liner
        max_peakind = np.pad(max_peakind, (1, 10), 'constant', constant_values=(0, 0))
        max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]

        for start in xrange(0, len(max_peakind) - 2):
            error = 10000000
            offset = 0
            for ii in xrange(0, 20):
                temp_error = model.get_left_leg().hip.angle.x[max_peakind[start + 1] + ii]
                if temp_error < error:
                    error = temp_error
                    offset = ii
            offsets.append(offset)

        for ii, start in enumerate(xrange(0, len(max_peakind) - 2)):
            begin = max_peakind[start]
            end = max_peakind[start + 1] + offsets[ii]
            # TODO need to find out if i need to round up or down
            vicon.append((begin, end))
            exo.append((int(math.ceil(theta * begin)), int(math.ceil(theta * end))))

        self.vicon_set_points = vicon  # varible that holds the setpoints for the vicon
        self.exo_set_points = exo  # varible that holds the setpoints for the exo

    def get_force_plates(self):
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
            if self._use_black_list:
                if key in self._black_list:
                    continue
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

    def get_joint_trajectories(self):
        """
        Seperates then joint trajs data
        :return: joint trajectory data
        :rtype: Dict
        """
        joints = {}
        count = 0
        model = self.vicon.get_model_output()
        for fnc, side in zip((model.get_left_leg(), model.get_right_leg()), ("R", "L")):
            for joint_name in ["hip", "knee", "ankle"]:
                name = side + joint_name
                joints[name] = []
                for inc in self.vicon_set_points:
                    time = ((inc[1] - inc[0]) / float(self.vicon.length)) * self.dt
                    time = np.linspace(0, 1, (inc[1] - inc[0]))
                    angle = core.Data(np.array(fnc._asdict()[joint_name].angle.x[inc[0]:inc[1]]), time)
                    power = core.Data(np.array(fnc._asdict()[joint_name].power.z[inc[0]:inc[1]]), time)
                    torque = core.Data(np.array(fnc._asdict()[joint_name].moment.x[inc[0]:inc[1]]), time)
                    force = core.Data(np.array(fnc._asdict()[joint_name].force.x[inc[0]:inc[1]]), time)
                    stamp = core.Newton(angle,force,torque,power)
                    if self._use_black_list:
                        if count in self._black_list:
                            continue
                    joints[name].append(stamp)
                    count+=1

        return joints

    def get_emg(self):
        """
       Seperates then EMGs data
       :return: EMGs data
       :rtype: Core.side
        """
        joints = {}
        count = 0
        emgs = self.vicon.get_all_emgs()

        for key, emg in emgs.iteritems():
            joints[key] = []
            for inc in self.vicon_set_points:
                start = emg.get_offset_index(inc[0])
                end = emg.get_offset_index(inc[1])
                data = np.array(emg.get_values())[start:end]
                time = (len(data) / float(self.vicon.length)) * self.dt
                stamp = core.Data(data, np.linspace(0, time, len(data)))
                if self._use_black_list:
                    if count in self._black_list:
                        continue
                joints[key].append(stamp)

                count += 1

        return joints

    def get_T_emgs(self):
        """
       Seperates then EMGs data
       :return: EMGs data
       :rtype: Core.side
        """
        joints = {}
        count = 0
        emgs = self.vicon.get_all_t_emg()

        for key, emg in emgs.iteritems():
            joints[key] = []
            for inc in self.vicon_set_points:
                start = emg.get_offset_index(inc[0])
                end = emg.get_offset_index(inc[1])
                data = np.array(emg.get_values())[start:end]
                time = (len(data) / float(self.vicon.length)) * self.dt
                stamp = core.Data(data, np.linspace(0, time, len(data)))
                if self._use_black_list:
                    if count in self._black_list:
                        continue
                joints[key].append(stamp)
                count += 1

        return joints

    def get_CoPs(self):
        """
       Seperates then CoP data
       :return: CoP data
       :rtype: Dict
        """

        left = []
        right = []
        count = 0
        left_cop = self.exoskeleton.left_leg.calc_CoP()
        right_cop = self.exoskeleton.right_leg.calc_CoP()

        left = []
        right = []

        for inc in self.exo_set_points:
            left_data = left_cop[inc[0]:inc[1]]
            right_data = right_cop[inc[0]:inc[1]]

            time = (len(left_data) / float(self.exoskeleton.length)) * self.dt
            stamp_left = core.Data(left_data, np.linspace(0, time, len(left_data)))
            stamp_right = core.Data(right_data, np.linspace(0, time, len(right_data)))

            if self._use_black_list:
                if count in self._black_list:
                    continue
                else:
                    left.append(stamp_left)
                    right.append(stamp_right)

            count += 1

        side = core.Side(left, right)

        return side

    def get_FSRs(self):
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

            # if self._use_black_list:
            #     if count in self._black_list:
            #         continue
            #     else:
            #         left.append(stamp_left)
            #         right.append(stamp_right)
            #
            # count += 1

        side = core.Side(left, right)

        return side

    def get_pots(self):
        """
       Seperates Pot data
       :return: Pot data
       :rtype: Dict
        """
        left_leg = self.exoskeleton.left_leg
        right_leg = self.exoskeleton.right_leg

        left = []
        right = []
        count = 0

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

            if self._use_black_list:
                if count in self._black_list:
                    continue
                else:
                    left.append(stamp_left)
                    right.append(stamp_right)
            count+=1

        side = core.Side(left, right)
        return side

    def get_accels(self):
        """
                Seperates then force plate data
                :return: Force plate data
                :rtype: Dict
                """
        left_leg = self.exoskeleton.left_leg
        right_leg = self.exoskeleton.right_leg

        left = []
        right = []
        count = 0
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

            if self._use_black_list:
                if count in self._black_list:
                    continue
                else:
                    left.append(stamp_left)
                    right.append(stamp_right)
            count+=1

        side = core.Side(left, right)

        return side

    def get_gyros(self):
        """
                Seperates then force plate data
                :return: Force plate data
                :rtype: Dict
                """
        left_leg = self.exoskeleton.left_leg
        right_leg = self.exoskeleton.right_leg

        left = []
        right = []
        count = 0

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

            if self._use_black_list:
                if count in self._black_list:
                    continue
                else:
                    left.append(stamp_left)
                    right.append(stamp_right)
            count += 1

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


    def add_to_blacklist(self, black_indexs):
        """
        Add a  blacklist
        :param index: index to add
        :return:
        """
        self._use_black_list = True
        self._black_list = black_indexs

    def remove_from_blacklist(self):
        """
        Remove the blacklist
        :param index: index to add
        :return:
        """
        self._use_black_list = False
        self._black_list = []

def calc_kinematics(trajectory, dt = 0.01):

    # y = trajectory.data
    # time = trajectory.time
    T = []
    y = trajectory
    #dt = time[1] - time[0]
    yp = [0.0]
    ypp = [0.0, 0.0]
    yp = np.append(yp, np.divide(np.diff(y, 1), np.power(dt, 1)))
    ypp = np.append(ypp, np.divide(np.diff(y, 2), np.power(dt, 2)))

    T.append(np.array(y))
    T.append(np.array(yp))
    T.append(np.array(ypp))

    return T

    # def plot(self):
    #
    #     plotter = TrialExaminer.TrialExaminer()
    #     joints = self.get_joint_trajectories()
    #     plates = self.get_force_plates()
    #     cops = self.get_CoPs()
    #     emgs = self.get_emgs()
    #
    #     accel = self.robot.get_accel
    #     gyro = self.robot.get_gyro
    #     pot = self.robot.get_pot
    #     fsr = self.robot.get_fsr
    #     left_fsr = [fsr["FSR1_Left"], fsr["FSR2_Left"], fsr["FSR3_Left"]]
    #     right_fsr = [fsr["FSR1_Right"], fsr["FSR2_Right"], fsr["FSR3_Right"]]
    #
    #     for key, sensor in accel.items():
    #         accel = PT.Line_Graph.Line_Graph(sensor.name, sensor, 3, ["x", "y", "z"])
    #         plotter.addfig(accel)
    #
    #     for key, sensor in gyro.items():
    #         gyro = PT.Line_Graph.Line_Graph(sensor.name, sensor, 3, ["x", "y", "z"])
    #         plotter.addfig(gyro)
    #
    #     for key, sensor in pot.items():
    #         pot = PT.Line_Graph.Line_Graph(sensor.name, sensor, 1, ["z"])
    #         plotter.addfig(pot)
    #
    #     fsr_plot = PT.FSR_BarGraph.FSR_BarGraph("FSR", fsr.values())
    #     plotter.addfig(fsr_plot)
    #     #
    #     cop_plot = PT.CoP_Plotter.CoP_Plotter("CoP", left_fsr, right_fsr)
    #     plotter.addfig(cop_plot)


if __name__ == '__main__':
    vicon_file = "/home/nathaniel/git/Gait_Analysis_Toolkit/Utilities/Walking01.csv"
    config_file = "/home/nathaniel/git/exoserver/Config/sensor_list.yaml"
    exo_file = "/home/nathaniel/git/exoserver/Main/subject_1234_trial_1.csv"
    trial = Trial(vicon_file, config_file, exo_file)
    joints = trial.seperate_joint_trajectories()
    plate = trial.seperate_force_plates()
    left, right = trial.seperate_CoP()

