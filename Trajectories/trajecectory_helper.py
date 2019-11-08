import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lib.dmp_experiments.Python import train_rmp, RMP_runner
from scipy.signal import find_peaks_cwt
import pandas

def sperate_joints(file_path):

    gait_data = pd.read_csv(file_path)
    names = ["LHipAngles", "LKneeAngles", "LAbsAnkleAngle", "RHipAngles", "RKneeAngles", "RAbsAnkleAngle"]
    offsets = []
    joints = {}

    hip = np.array(gait_data["LHipAngles"]).flatten()
    N = 10
    hip = np.convolve(hip, np.ones((N,)) / N, mode='valid')

    max_peakind = np.diff(np.sign(np.diff(hip))).flatten() #the one liner
    max_peakind = np.pad(max_peakind, (1, 10), 'constant', constant_values=(0, 0))
    max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]



    for start in xrange(0, len(max_peakind) -1 ):
        error = 10000000
        offset = 0
        for ii in xrange(0, 25):
            temp_error = gait_data["LKneeAngles"][max_peakind[start+1] + ii]
            if temp_error < error:
                error = temp_error
                offset = ii
        offsets.append(offset)

    for name in names:
        joints[name] = []
        for ii, start in enumerate(xrange(0, len(max_peakind) - 1 )):
            joints[name].append(np.array(gait_data[name][max_peakind[start]:max_peakind[start+1]+offsets[ii]]))

    return joints, hip


def generate_dmps(prefixes, data, index):

    train_rmp.train_rmp(prefixes + "_ankle_right.xml", 1000, np.radians(np.array([data["RAbsAnkleAngle"][index]])), 0.01)
    train_rmp.train_rmp(prefixes + "_knee_right.xml", 1000, np.radians(np.array([data["RKneeAngles"][index]])), 0.01)
    train_rmp.train_rmp(prefixes + "_hip_right.xml", 1000, np.radians(np.array([data["RHipAngles"][index]])), 0.01)

    train_rmp.train_rmp(prefixes + "_ankle_left.xml", 1000, np.radians(np.array([data["LAbsAnkleAngle"][index]])), 0.01)
    train_rmp.train_rmp(prefixes + "_knee_left.xml", 1000, np.radians(np.array([data["LKneeAngles"][index]])), 0.01)
    train_rmp.train_rmp(prefixes + "_hip_left.xml", 1000, np.radians(np.array([data["LHipAngles"][index]])), 0.01)


if __name__ == "__main__":
    joints, data = sperate_joints("/home/nathaniel/Desktop/test_joint_angles.csv")
    #joints, data = sperate_joints("/home/nathaniel/catkin_ws/src/AMBF_Walker/config/joint_data_edited.csv")
    print joints
    for joint in joints["RHipAngles"]:
        plt.plot(joint)
    #plt.plot(data)
    plt.show()