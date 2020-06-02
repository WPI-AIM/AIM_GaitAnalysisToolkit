import sys
import matplotlib.pyplot as plt
import lib.GaitCore.Core.utilities as utilities
from scipy import signal
import numpy as np
from Session import ViconGaitingTrial
from lib.Vicon import Vicon
import os
from scipy import signal
import numpy.polynomial.polynomial as poly
from dtw import dtw

def refit(x, y):
    manhattan_distance = lambda x, y: np.abs(x - y)

    d, cost_matrix, acc_cost_matrix, path = dtw(x, y, dist=manhattan_distance)
    data_warp = [y[path[1]][:x.shape[0]]]
    t = np.linspace(0, 1.0, len(data_warp[0]))
    coefs = poly.polyfit(t, data_warp[0], 20)
    ffit = poly.Polynomial(coefs)  # instead of np.poly1d
    y_fit = ffit(t)
    y_fit = data_warp[0]
    temp = [[np.array(ele)] for ele in y_fit.tolist()]
    return np.array(temp)



def plot_joint_angles(files, indecies, sides, lables):
    """

    :param files:
    :param indecies:
    :param sides:
    :param lables:
    :return:
    """
    angles = {}
    angles["hip"] = []
    angles["knee"] = []
    angles["ankle"] = []
    samples = []
    for file, i, side in zip(files, indecies, sides):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        trial.create_index_seperators()
        body = trial.get_joint_trajectories()
        if side == "L":
            hip_angle = body.left.hip[i].angle.x.data
            knee_angle = body.left.knee[i].angle.x.data
            ankle_angle = body.left.ankle[i].angle.x.data
        else:
            hip_angle = body.right.hip[i].angle.x.data
            knee_angle = body.right.knee[i].angle.x.data
            ankle_angle = body.right.ankle[i].angle.x.data

        angles["hip"].append(hip_angle)
        angles["knee"].append(knee_angle)
        angles["ankle"].append(ankle_angle)
        samples.append(len(hip_angle))

    plt.figure(1)
    f, (ax1_angle, ax2_angle, ax3_angle) = plt.subplots(3, 1)
    sample_size = max(samples)
    index = samples.index(sample_size)

    for i in range(len(files)):
        ax1_angle.plot(angles["hip"][i])
        ax2_angle.plot(angles["knee"][i])
        ax3_angle.plot(angles["ankle"][i])

    plt.show()


if __name__ == "__main__":
    plot_joint_angles(["/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 walk_02.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 walk_01.csv",
                       ],
                      [2, 1, 0, 2, 0],
                      ["L", "L", "L", "L", "L"],
                      ["Subject10", "Subject00"])



    ["/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 walk_00.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_01/subject_01_walk_00.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_walk_00.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_walk_00.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_04/subject_04_walk_00.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_05/subject_05_walk_00.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 walk_00.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 walk_00.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_walking_01.csv",
     "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 walk_00.csv"]

    [1, 8, 16, 2, 11, 4, 4, 11, 16, 9, 9]

    ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7", "Subject8",
     "Subject10"]
