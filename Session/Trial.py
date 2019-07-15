import matplotlib.pyplot as plt

import numpy as np

from Vicon import Vicon


class Trial(object):

    def __init__(self, vicon_file, config_file=None, exo_file=None, dt=None, notes_file=None):

        # self._notes_file = notes_file
        self._dt = 100
        # self._exoskeleton = Exoskeleton.Exoskeleton(config_file, exo_file)
        self._vicon = Vicon.Vicon(vicon_file)
        self._joint_trajs = None
        self._black_list = []


    def get_index_seperators(self):
        names = ["HipAngles", "KneeAngles", "AbsAnkleAngle"]
        offsets = []
        model = self.vicon.get_model_output()
        hip = model.get_right_joint("RHipAngles").angle.x

        max_peakind = np.diff(np.sign(np.diff(hip))).flatten()  # the one liner
        max_peakind = np.pad(max_peakind, (1, 10), 'constant', constant_values=(0, 0))
        max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]

        for start in xrange(2, len(max_peakind) - 2):
            error = 10000000
            offset = 0
            for ii in xrange(0, 25):
                temp_error = model.get_left_joint("LKneeAngles").angle.x[max_peakind[start + 1] + ii]
                if temp_error < error:
                    error = temp_error
                    offset = ii
            offsets.append(offset)

        for side in ("R", "L"):
            for joint_name in names:
                name = side + joint_name
                joints[name] = []
                for ii, start in enumerate(xrange(2, len(max_peakind) - 2)):
                    data = (max_peakind[start],max_peakind[start + 1] + offsets[ii])
                    joints[name].append((data, np.linspace(0, self.dt, len(data))))

    def separate_joint_trajectories(self):

        joints = {}
        names = ["HipAngles", "KneeAngles", "AbsAnkleAngle"]
        offsets = []
        model = self.vicon.get_model_output()
        hip = model.get_right_joint("RHipAngles").angle.x
        start = np.argmax(np.array(hip) > 0)
        dH = np.gradient(hip)

        max_peakind = np.diff(np.sign(np.diff(hip))).flatten()  # the one liner
        max_peakind = np.pad(max_peakind, (1, 10), 'constant', constant_values=(0, 0))
        max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]

        for start in xrange(2, len(max_peakind) - 2):
            error = 10000000
            offset = 0
            for ii in xrange(0, 25):
                temp_error = model.get_left_joint("LKneeAngles").angle.x[max_peakind[start + 1] + ii]
                if temp_error < error:
                    error = temp_error
                    offset = ii
            offsets.append(offset)

        for fnc, side in zip((model.get_left_joint, model.get_right_joint), ("R", "L")):
            for joint_name in names:
                name = side + joint_name
                joints[name] = []
                for ii, start in enumerate(xrange(2, len(max_peakind) - 2)):
                    data = np.array(fnc(name).angle.x[max_peakind[start]:max_peakind[start + 1] + offsets[ii]])
                    joints[name].append((data, np.linspace(0, self.dt, len(data))))

        self._joint_trajs = joints

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
    file = "/home/nathaniel/git/Gait_Analysis_Toolkit/Utilities/Walking01.csv"
    trial = Trial(file)
    trial.separate_joint_trajectories()
    joints = trial.joint_trajs
    fig, ax = plt.subplots()
    index = 0
    print joints
    for index in xrange(1):
        y = joints["RKneeAngles"][index][0]
        x = joints["RKneeAngles"][index][1]
        ax.plot(x,y)

    plt.show()
