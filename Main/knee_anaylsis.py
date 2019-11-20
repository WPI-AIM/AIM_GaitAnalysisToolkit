import sys
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from lib.dmp_experiments.Python import train_dmp, DMP_runner
from Session import Trial
from Vicon import Vicon


def plot_leg_joints(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking data')
    hip = []
    knee = []
    ankle = []
    time = None
    resample = 100000
    for file, i in zip(files, list_of_index):
        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):

        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip.append( signal.resample( joints["Rhip"][i].angle.data, resample))
        knee.append(signal.resample( joints["Rknee"][i].angle.data, resample))
        ankle.append(signal.resample( joints["Rankle"][i].angle.data, resample))


    time = np.linspace(0,1, resample)
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    mean_hip = np.mean(hip, axis=0)
    mean_knee = np.mean(knee, axis=0)
    mean_ankle = np.mean(ankle, axis=0)

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)

    ax1.plot(time, mean_hip, 'k-')
    ax2.plot(time, mean_knee, 'k-')
    ax3.plot(time, mean_ankle, 'k-')

    ax1.fill_between(time, mean_hip - std_hip, mean_hip + std_hip)
    ax2.fill_between(time, mean_knee - std_knee, mean_knee + std_knee)
    ax3.fill_between(time, mean_ankle - std_ankle, mean_ankle + std_ankle)

    ax1.set_ylabel("Degrees")
    ax2.set_ylabel("Degrees")
    ax3.set_ylabel("Degrees")
    ax1.set_title("Hip")
    ax2.set_title("Knee")
    ax3.set_title("Ankle")
    plt.xlabel("Gait %")

    plt.show()


def plot_signle_knee(file):
    trial = Trial.Trial(vicon_file="/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_00/11_07_2019/subject_00 walk_01.csv")
    trial = Trial.Trial(vicon_file=file)
    joints = trial.get_joint_trajectories()
    leg = []
    for i in xrange(len(joints["Rknee"])):
        leg.append(i)
        plt.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].angle.data)
    plt.legend(leg)
    plt.show()

def plot_knee(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking data')

    for file, i in zip(files,list_of_index):
        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        ax1.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].angle.data)
        ax2.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].moment.data)
        ax3.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].power.data)



    if legend:
        ax1.legend(legend)
    ax1.set_ylabel("Degrees")
    ax2.set_ylabel("Nmm/Kg")
    ax3.set_ylabel("W/Kg")
    ax1.set_title("Angle")
    ax2.set_title("Torque")
    ax3.set_title("Power")
    plt.xlabel("Gait %")

    plt.show()

if __name__ == "__main__":
    # plot_knee(["/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_00/11_07_2019/subject_00 walk_00.csv",
    #            "/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_03/11_8_2019/subject_03_walk_00.csv"],
    #           [1, 22],
    #           ["Subject0", "Subject1"])
    plot_leg_joints(["/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_00/11_07_2019/subject_00 walk_00.csv",
               "/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_03/11_8_2019/subject_03_walk_00.csv"],
              [1, 22],
              ["Subject0", "Subject1"])

    # plot_data(["/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_03/11_8_2019/subject_03_walk_00.csv"],
    #           [22],
    #           ["Subject1"])
    #plot_signle_knee("/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_03/11_8_2019/subject_03_walk_00.csv")