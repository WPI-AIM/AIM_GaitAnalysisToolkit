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
    hip = []
    knee = []
    ankle = []
    time = None
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
        hip.append( signal.resample( joints["Rhip"][i].angle.data, resample))
        knee.append(signal.resample( joints["Rknee"][i].angle.data, resample))
        ankle.append(signal.resample( joints["Rankle"][i].angle.data, resample))

    # make into arrays and smooth
    time = np.linspace(0,1, resample)
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    mean_hip = utilities.smooth(np.mean(hip, axis=0), 5)
    mean_knee = utilities.smooth(np.mean(knee, axis=0), 5)
    mean_ankle = utilities.smooth(np.mean(ankle, axis=0), 5)

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)

    print "Ankle: "
    print "Max Hip: ", np.max(np.abs(mean_hip)), " Std: ", std_hip[mean_hip.tolist().index(np.max(mean_hip))]
    print "Max Knee: ", np.max(np.abs(mean_knee)), " Std: ", std_knee[mean_knee.tolist().index(np.max(mean_knee))]
    print "Max Ankle: ", np.max(np.abs(mean_ankle)), " Std: ", std_ankle[mean_ankle.tolist().index(np.max(mean_ankle))]

    print "Min Hip: ", np.min(np.abs(mean_hip)), " Std: ", std_hip[mean_hip.tolist().index(np.min(mean_hip))]
    print "Min Knee: ", np.min(np.abs(mean_knee)), " Std: ", std_knee[mean_knee.tolist().index(np.min(mean_knee))]
    print "Min Ankle: ", np.min(np.abs(mean_ankle)), " Std: ", std_ankle[mean_ankle.tolist().index(np.min(mean_ankle))]


    # plot everything
    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)


    ax1.fill_between(time, utilities.smooth(mean_hip - std_hip, 5), utilities.smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, utilities.smooth(mean_knee - std_knee, 5), utilities.smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, utilities.smooth(mean_ankle - std_ankle, 5), utilities.smooth(mean_ankle + std_ankle, 5))
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
    fig.suptitle('Walking Joint Moments', fontsize=20)
    hip = []
    knee = []
    ankle = []
    time = None
    resample = 100000
    for file, i in zip(files, list_of_index):
        print "i ", i
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        print joints["Rhip"]
        print "file ", file
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):

        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip.append( signal.resample(  abs(joints["Rhip"][i].moment.data), resample))
        knee.append(signal.resample(abs( joints["Rknee"][i].moment.data), resample))
        ankle.append(signal.resample( abs(joints["Rankle"][i].moment.data), resample))


    time = np.linspace(0,1, resample)
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    mean_hip = utilities.smooth(np.mean(hip, axis=0), 5)
    mean_knee = utilities.smooth(np.mean(knee, axis=0), 5)
    mean_ankle = utilities.smooth(np.mean(ankle, axis=0), 5   )

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)


    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)


    ax1.fill_between(time, utilities.smooth(mean_hip - std_hip, 5), utilities.smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, utilities.smooth(mean_knee - std_knee, 5), utilities.smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, utilities.smooth(mean_ankle - std_ankle, 5), utilities.smooth(mean_ankle + std_ankle, 5))
    font_size = 25
    ax1.set_ylabel("Nmm/KG", fontsize=font_size)
    ax2.set_ylabel("Nmm/KG", fontsize=font_size)
    ax3.set_ylabel("Nmm/KG", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()

if __name__ == "__main__":

    cur_path = os.path.dirname(__file__)
    print cur_path
    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in
    rel_path = "ExampleData/subject_00 walk_00.csv"
    abs_file_path = os.path.join(script_dir, rel_path)
    print abs_file_path

    compare_walking_angles(
        [os.path.join(script_dir, "ExampleData/subject_00 walk_00.csv"),
         os.path.join(script_dir, "ExampleData/subject_01 walk_00.csv")],
        [1, 8])

    compare_walking_moments(
        [os.path.join(script_dir, "ExampleData/subject_00 walk_00.csv"),
         os.path.join(script_dir, "ExampleData/subject_01 walk_00.csv")],
        [1, 8])
