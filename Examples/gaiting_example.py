import sys
import matplotlib.pyplot as plt
import lib.GaitCore.Core.utilities as utilities
from scipy import signal
import numpy as np
from Session import ViconGaitingTrial
from lib.Vicon import Vicon
import os



"""
This shows how to use the gaiting tools, it seperates the joint trajectores and 
plots the joint angle and moments

"""


def compare_walking_angles(files, list_of_index):
    """
    Plots the joint angle during walking
    :param files: sting of file name
    :param list_of_index: which traj to use
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Angles', fontsize=20)
    resample = 100000

    # find resample length to use
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    # grab all the trajs and resample
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip = signal.resample( joints["Rhip"][i].angle.data, resample)
        knee = signal.resample(joints["Rknee"][i].angle.data, resample)
        ankle = signal.resample(joints["Rankle"][i].angle.data, resample)
        ax1.plot(hip)
        ax2.plot(knee)
        ax3.plot(ankle)


    font_size = 25
    ax1.set_ylabel("Degrees", fontsize=font_size)
    ax2.set_ylabel("Degrees", fontsize=font_size)
    ax3.set_ylabel("Degrees", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()


def compare_walking_moments(files, list_of_index):
    """
      Plots the joint angle during walking
      :param files: sting of file name
      :param list_of_index: which traj to use
      """
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Angles', fontsize=20)
    resample = 100000

    # find resample length to use
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        sample = len(joints["Rhip"][i].moment.data)
        resample = min(resample, sample)

    # grab all the trajs and resample
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip = signal.resample(joints["Rhip"][i].moment.data, resample)
        knee = signal.resample(joints["Rknee"][i].moment.data, resample)
        ankle = signal.resample(joints["Rankle"][i].moment.data, resample)
        ax1.plot(utilities.smooth(hip, 10))
        ax2.plot(utilities.smooth(knee, 10))
        ax3.plot( utilities.smooth(ankle,10))

    font_size = 25
    ax1.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax2.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax3.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)


    plt.show()


def compare_stair_angles(files, side, legend):

    resample = 100000
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Stair Joint Angles', fontsize=20)

    indiecs = {}
    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(file)
        rn = trial.get_stair_ranges(s)
        indiecs[file] = rn
        resample = min(resample, rn[1] - rn[0])

    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        if s == "R":
            joints = trial.vicon.get_model_output().right_leg()
        else:
            joints = trial.vicon.get_model_output().left_leg()
        rn = indiecs[file]
        hip = signal.resample(joints.hip.angle.x[rn[0]:rn[1]], resample)
        knee = signal.resample(joints.knee.angle.x[rn[0]:rn[1]], resample)
        ankle = signal.resample(joints.ankle.angle.x[rn[0]:rn[1]], resample)
        ax1.plot(hip)
        ax2.plot(knee)
        ax3.plot(ankle)

    plt.legend(legend)
    font_size = 25
    ax1.set_ylabel("Degrees", fontsize=font_size)
    ax2.set_ylabel("Degrees", fontsize=font_size)
    ax3.set_ylabel("Degrees", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()


def compare_stair_moments(files, side, legend):
    resample = 100000
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Stair Joint Moments', fontsize=20)

    resample = 1000000000

    indiecs = {}
    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(file)
        rn = trial.get_stair_ranges(s)
        indiecs[file] = rn
        resample = min(resample, rn[1] - rn[0])

    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        if s == "R":
            joints = trial.vicon.get_model_output().right_leg()
        else:
            joints = trial.vicon.get_model_output().left_leg()
        rn = indiecs[file]
        hip = signal.resample(joints.hip.angle.x[rn[0]:rn[1]], resample)
        knee = signal.resample(joints.knee.angle.x[rn[0]:rn[1]], resample)
        ankle = signal.resample(joints.ankle.angle.x[rn[0]:rn[1]], resample)
        ax1.plot(utilities.smooth(hip,10))
        ax2.plot( utilities.smooth(knee,10))
        ax3.plot(utilities.smooth(ankle,10))

    #plt.legend(legend)
    font_size = 25
    ax1.set_ylabel("Nmm/kg", fontsize=font_size)
    ax2.set_ylabel("Nmm/kg", fontsize=font_size)
    ax3.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()




if __name__ == "__main__":
    pass
    # script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in


    # compare_walking_angles(
    #     [os.path.join(script_dir, "ExampleData/subject_00 walk_00.csv"),
    #      os.path.join(script_dir, "ExampleData/subject_01 walk_00.csv")],
    #     [1, 8])
    #
    # compare_walking_moments(
    #     [os.path.join(script_dir, "ExampleData/subject_00 walk_00.csv"),
    #      os.path.join(script_dir, "ExampleData/subject_01 walk_00.csv")],
    #     [1, 8])


    # compare_walking_angles(
    #     ["/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_01/subject_01 walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_02/subject_02_walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_03/subject_03_walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_04/subject_04_walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_05/subject_05_walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_07/subject_07 walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_08/subject_08_walking_01.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_09/subject_09 walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_10/subject_10 walk_00.csv"],
    #     [1, 8, 16, 2, 11, 4,  11, 16, 9, 9]
    #     )
    #
    # compare_walking_moments(
    #     ["/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_01/subject_01 walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_02/subject_02_walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_03/subject_03_walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_04/subject_04_walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_05/subject_05_walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_07/subject_07 walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_08/subject_08_walking_01.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_09/subject_09 walk_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_10/subject_10 walk_00.csv"],
    #     [1, 8, 16, 2, 11, 4, 11, 16, 9, 9 ]
    # )

    # compare_stair_angles(
    #     ["/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_00/subject_00 stairconfig1_01.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_02/subject_02_stair_config1_03.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_03/subject_03_stair_config0_02.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_04/subject_04_stair_config1_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_05/subject_05_stair_config1_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_06/subject_06 stairclimbing_config1_02.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_07/subject_07 stairclimbing_config1_01.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_09/subject_09 stairclimbing_config1_00.csv" ],
    #     ["R", "R", "L", "L", "R", "L", "R", "R"],
    #     ["subject00", "subject02", "Subject03", "Subject04", "Subject05", "Subject06", "Subject07",
    #      "Subject09" ])
    #
    # compare_stair_moments(
    #     ["/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_00/subject_00 stairconfig1_01.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_02/subject_02_stair_config1_03.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_03/subject_03_stair_config0_02.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_04/subject_04_stair_config1_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_05/subject_05_stair_config1_00.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_06/subject_06 stairclimbing_config1_02.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_07/subject_07 stairclimbing_config1_01.csv",
    #      "/media/nathanielgoldfarb/New Volume/stairclimbing_data/CSVs/subject_09/subject_09 stairclimbing_config1_00.csv" ],
    #     ["R", "R", "L", "L", "R", "L", "R", "R"],
    #     ["subject00", "subject02", "Subject03", "Subject04", "Subject05", "Subject06", "Subject07",
    #      "Subject09" ])
