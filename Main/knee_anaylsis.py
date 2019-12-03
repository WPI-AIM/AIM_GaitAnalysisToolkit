import sys
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from lib.dmp_experiments.Python import train_dmp, DMP_runner
from Session import Trial
from Vicon import Vicon



def plot_leg_power(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Power')
    hip = []
    knee = []
    ankle = []
    time = None
    resample = 100000
    for file, i in zip(files, list_of_index):
        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        print "file ", file
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):

        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip.append( signal.resample( joints["Rhip"][i].power.data, resample))
        knee.append(signal.resample( joints["Rknee"][i].power.data, resample))
        ankle.append(signal.resample( joints["Rankle"][i].power.data, resample))


    time = np.linspace(0,1, resample)
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    mean_hip = smooth(np.mean(hip, axis=0), 5)
    mean_knee = smooth(np.mean(knee, axis=0), 5)
    mean_ankle = smooth(np.mean(ankle, axis=0), 5   )

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)

    ax1.plot(time, mean_hip, 'k-')
    ax2.plot(time, mean_knee, 'k-')
    ax3.plot(time, mean_ankle, 'k-')

    ax1.fill_between(time, mean_hip - std_hip, mean_hip + std_hip)
    ax2.fill_between(time, mean_knee - std_knee, mean_knee + std_knee)
    ax3.fill_between(time, mean_ankle - std_ankle, mean_ankle + std_ankle)

    ax1.set_ylabel("W/KG")
    ax2.set_ylabel("W/KG")
    ax3.set_ylabel("W/KG")
    ax1.set_title("Hip")
    ax2.set_title("Knee")
    ax3.set_title("Ankle")
    plt.xlabel("Gait %")

    plt.show()



def plot_leg_moments(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Moments')
    hip = []
    knee = []
    ankle = []
    time = None
    resample = 100000
    for file, i in zip(files, list_of_index):
        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        print "file ", file
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):

        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip.append( signal.resample( joints["Rhip"][i].moment.data, resample))
        knee.append(signal.resample( joints["Rknee"][i].moment.data, resample))
        ankle.append(signal.resample( joints["Rankle"][i].moment.data, resample))


    time = np.linspace(0,1, resample)
    hip = np.array(hip)
    knee = np.array(knee)
    ankle = np.array(ankle)

    mean_hip = smooth(np.mean(hip, axis=0), 5)
    mean_knee = smooth(np.mean(knee, axis=0), 5)
    mean_ankle = smooth(np.mean(ankle, axis=0), 5   )

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)

    ax1.plot(time, mean_hip, 'k-')
    ax2.plot(time, mean_knee, 'k-')
    ax3.plot(time, mean_ankle, 'k-')

    ax1.fill_between(time, mean_hip - std_hip, mean_hip + std_hip)
    ax2.fill_between(time, mean_knee - std_knee, mean_knee + std_knee)
    ax3.fill_between(time, mean_ankle - std_ankle, mean_ankle + std_ankle)

    ax1.set_ylabel("Nmm/KG")
    ax2.set_ylabel("Nmm/KG")
    ax3.set_ylabel("Nmm/KG")
    ax1.set_title("Hip")
    ax2.set_title("Knee")
    ax3.set_title("Ankle")
    plt.xlabel("Gait %")

    plt.show()


def plot_leg_joints(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Angles')
    hip = []
    knee = []
    ankle = []
    time = None
    resample = 100000
    for file, i in zip(files, list_of_index):
        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        print "file ", file
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

    mean_hip = smooth(np.mean(hip, axis=0), 5)
    mean_knee = smooth(np.mean(knee, axis=0), 5)
    mean_ankle = smooth(np.mean(ankle, axis=0), 5   )

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
    max_joint = []
    min_joint = []
    max_power = []
    min_power = []
    max_moment = []
    min_moment = []
    for file, i in zip(files,list_of_index):
        trial = Trial.Trial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        max_joint.append(max(abs(joints["Rknee"][i].angle.data)))
        min_joint.append(min(abs(joints["Rknee"][i].angle.data)))
        max_power.append(max(abs(joints["Rknee"][i].power.data)))
        min_power.append(min(abs(joints["Rknee"][i].power.data)))
        max_moment.append(max(abs(joints["Rknee"][i].moment.data)))
        min_moment.append(min(abs(joints["Rknee"][i].moment.data)))
        ax1.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].angle.data)
        ax2.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].moment.data)
        ax3.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].power.data)

    print "Angle: "
    print "Max Mean: ", np.mean(max_joint), " Std: ", np.std(max_joint)
    print "Min Mean: ", np.mean(min_joint), " Std: ", np.std(min_joint)

    print "Moment: "
    print "Max Mean: ", np.mean(max_moment), " Std: ", np.std(max_moment)

    print "Power: "
    print "Max Mean: ", np.mean(max_power), " Std: ", np.std(max_power)

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

def plot_stairs(files):
    pass


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == "__main__":
    plot_leg_power(["/home/nathaniel/Documents/MoCap_Participants/subject_00/subject_00_walk_00.csv",
                     "/home/nathaniel/Documents/MoCap_Participants/subject_01/subject_01_walk_00.csv",
                     "/home/nathaniel/Documents/MoCap_Participants/subject_02/subject_02_walk_00.csv",
                     "/home/nathaniel/Documents/MoCap_Participants/subject_03/subject_03_walk_00.csv",
                     "/home/nathaniel/Documents/MoCap_Participants/subject_04/subject_04_walk_00.csv",
                      "/home/nathaniel/Documents/MoCap_Participants/subject_05/subject_05_walk_00.csv",
                      "/home/nathaniel/Documents/MoCap_Participants/subject_06/subject_06_walk_00.csv",
                      "/home/nathaniel/Documents/MoCap_Participants/subject_07/subject_07_walk_00.csv"],
                     [1, 8, 16, 2, 11, 4, 4, 11],
                     ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7"])

    # plot_leg_joints(["/home/nathaniel/Documents/MoCap_Participants/subject_00/subject_00_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_01/subject_01_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_02/subject_02_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_03/subject_03_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_04/subject_04_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_05/subject_05_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_06/subject_06_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_07/subject_07_walk_00.csv"],
    #           [1, 8, 16, 2, 11, 4, 4, 11 ],
    #           ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7"])

    # plot_knee(["/home/nathaniel/Documents/MoCap_Participants/subject_00/subject_00_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_01/subject_01_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_02/subject_02_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_03/subject_03_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_04/subject_04_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_05/subject_05_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_06/subject_06_walk_00.csv",
    #            "/home/nathaniel/Documents/MoCap_Participants/subject_07/subject_07_walk_00.csv"],
    #           [1, 8, 16, 2, 11, 4, 4, 11],
    #           ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7"])
    # plot_data(["/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_03/11_8_2019/subject_03_walk_00.csv"],
    #           [22],
    #           ["Subject1"])
    #plot_signle_knee("/home/nathaniel/Documents/MoCap_Participants/subject_07/subject_07_walk_00.csv")

    plot_knee(["/home/nathaniel/Documents/MoCap_Participants/subject_07/subject_07_walk_00.csv"],
              [10],
              [ "Subject7"])
