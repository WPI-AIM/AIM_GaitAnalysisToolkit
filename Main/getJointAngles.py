import sys
import matplotlib.pyplot as plt
import lib.GaitCore.Core.utilities as utilities
from scipy import signal
import numpy as np
from Session import ViconGaitingTrial
from lib.Vicon import Vicon
import os


def plot_joint_angles(files, indecies, lables):

    angles = {}
    moments = {}

    moments["hip"] = []
    moments["knee"] = []
    moments["ankle"] = []
    angles["hip"] = []
    angles["knee"] = []
    angles["ankle"] = []

    for file, i in zip(files, indecies):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        trial.create_index_seperators()
        body = trial.get_joint_trajectories()
        hip_angle = body.left.hip[i].angle.x.data
        knee_angle = body.left.knee[i].angle.x.data
        ankle_angle = body.left.ankle[i].angle.x.data

        hip_moment = body.left.hip[i].moment.x.data
        knee_moment = body.left.knee[i].moment.x.data
        ankle_moment = body.left.ankle[i].moment.x.data

        angles["hip"].append(hip_angle)
        angles["knee"].append(knee_angle)
        angles["ankle"].append(ankle_angle)

        moments["hip"].append(hip_moment)
        moments["knee"].append(knee_moment)
        moments["ankle"].append(ankle_moment)

    plt.figure(1)
    f, (ax1_angle, ax2_angle, ax3_angle) = plt.subplots(3, 1)
    ax1_angle.plot(angles["hip"][0])
    ax2_angle.plot(angles["knee"][0])
    ax3_angle.plot(angles["ankle"][0])

    plt.figure(2)
    f, (ax1_torque, ax2_torque, ax3_torque) = plt.subplots(3, 1)
    ax1_torque.plot(moments["hip"][0])
    ax2_torque.plot(moments["knee"][0])
    ax3_torque.plot(moments["ankle"][0])
    plt.show()


if __name__ == "__main__":
    plot_joint_angles(["/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 walk_02.csv"],
                      [2],
                      ["dfa"])

