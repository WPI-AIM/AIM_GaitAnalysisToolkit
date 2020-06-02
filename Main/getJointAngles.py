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
    SMALL_SIZE = 10
    MEDIUM_SIZE = 10
    BIGGER_SIZE = 18

    plt.rc('font', size=SMALL_SIZE)  # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc('axes', labelsize=BIGGER_SIZE)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('ytick', labelsize=BIGGER_SIZE)  # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
    f, (ax1_angle, ax2_angle, ax3_angle) = plt.subplots(3, 1)
    sample_size = min(samples)

    ax1_angle.set_title("Hip Angle")
    ax2_angle.set_title("Knee Angle")
    ax3_angle.set_title("Ankle Angle")
    f.suptitle('Walking Joint Angles', fontsize=20)

    ax1_angle.set_ylabel("Angle (Degrees)")
    ax2_angle.set_ylabel("Angle (Degrees)")
    ax3_angle.set_ylabel("Angle (Degrees)")
    t = np.linspace(0,100,sample_size)

    for i in range(len(files)):
        ax1_angle.plot(t, signal.resample(angles["hip"][i], sample_size))
        ax2_angle.plot(t, signal.resample(angles["knee"][i], sample_size))
        ax3_angle.plot(t, signal.resample(angles["ankle"][i], sample_size))

    plt.xlabel("Gait %")
    plt.legend(lables)

    plt.show()


if __name__ == "__main__":
    plot_joint_angles(["/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_00/subject_00 walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_01/subject_01_walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_02/subject_02_walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_03/subject_03_walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_04/subject_04_walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_05/subject_05_walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_06/subject_06 walk_00.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_07/subject_07 walk_01.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_08/subject_08_walking_01.csv",
                       "/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 walk_02.csv",
                       ],
                      [ 0, 0, 1, 0, 0, 0, 0, 0, 0, 2],
                      ["R", "R", "R", "R", "R", "R", "R", "R","R", "R" ],
                      ["Subject00", "Subject01", "Subject02",  'Subject03', "subject04", "Subject05", "Subject06", "Subject07",  "Subject08", "Subject10"])



