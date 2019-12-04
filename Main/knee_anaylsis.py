import sys
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from lib.dmp_experiments.Python import train_dmp, DMP_runner
from Session import Trial
from Vicon import Vicon


def plot_stair_joint(file):

    trial = Trial.Trial(vicon_file=file)
    joints = trial.vicon.get_model_output().get_right_leg()
    #trial.vicon.markers.play()
    #print joints.hip.angle.x
    plt.plot(joints.hip.angle.x)
    plt.plot(joints.knee.angle.x)
    plt.plot(joints.ankle.angle.x)
    plt.show()

def compare_stair_joints(files, ranges):

    resample = 100000
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    hip = []
    knee = []
    ankle = []
    resample = 1000000000

    for rn in ranges:
        resample = min(resample, abs(rn[0] - rn[1]))

    for file, i in zip(files, ranges):
        trial = Trial.Trial(vicon_file=file)
        joints = trial.vicon.get_model_output().get_right_leg()
        hip = signal.resample(joints.hip.angle.x[rn[0]:rn[1]], resample)
        knee = signal.resample(joints.knee.angle.x[rn[0]:rn[1]], resample)
        ankle = signal.resample(joints.ankle.angle.x[rn[0]:rn[1]], resample)
        ax1.plot(hip)
        ax2.plot(knee)
        ax3.plot(ankle)

    plt.show()
        # hip.append(signal.resample(joints.hip.angle[rn[0]:rn[1]], resample))
        # knee.append(signal.resample(joints.knee.angle[rn[0]:rn[1]], resample))
        # ankle.append(signal.resample(joints.ankle.angle[rn[0]:rn[1]], resample))






def plot_leg_power(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Power', fontsize=20)

    hip = []
    knee = []
    ankle = []
    max_joint = []
    min_joint = []
    power = []
    moment = []

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
    mean_ankle = smooth(np.mean(ankle, axis=0), 5)

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)

    print "Power: "
    print "Max Hip: ", np.max(np.abs(mean_hip)), " Std: ", std_hip[mean_hip.tolist().index(np.max(np.abs(mean_hip)))]
    print "Max Knee: ", np.max(np.abs(mean_knee)), " Std: ", std_knee[mean_knee.tolist().index(-np.max(np.abs(mean_knee)))]
    print "Max Ankle: ", np.max(np.abs(mean_ankle)), " Std: ", std_ankle[mean_ankle.tolist().index(-np.max(np.abs(mean_ankle)))]


    ax1.plot(time, mean_hip, 'k-')
    ax2.plot(time, mean_knee, 'k-')
    ax3.plot(time, mean_ankle, 'k-')

    ax1.fill_between(time, smooth( mean_hip - std_hip, 5), smooth(mean_hip + std_hip,5))
    ax2.fill_between(time, smooth(mean_knee - std_knee,5), smooth(mean_knee + std_knee,5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle,5), smooth(mean_ankle + std_ankle,5))

    ax1.set_ylabel("W/KG", fontsize=20),
    ax2.set_ylabel("W/KG", fontsize=20)
    ax3.set_ylabel("W/KG", fontsize=20)
    ax1.set_title("Hip", fontsize=20)
    ax2.set_title("Knee", fontsize=20)
    ax3.set_title("Ankle", fontsize=20)
    plt.xlabel("Gait %", fontsize=20)

    plt.show()



def plot_leg_moments(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Moments', fontsize=20)
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


    print "Moment: "
    print "Max Hip: ", np.max(np.abs(mean_hip)), " Std: ", std_hip[mean_hip.tolist().index(np.max(np.abs(mean_hip)))]
    print "Max Knee: ", np.max(np.abs(mean_knee)), " Std: ", std_knee[mean_knee.tolist().index(-np.max(np.abs(mean_knee)))]
    print "Max Ankle: ", np.max(np.abs(mean_ankle)), " Std: ", std_ankle[mean_ankle.tolist().index(np.max(np.abs(mean_ankle)))]

    ax1.plot(time, mean_hip, 'k-')
    ax2.plot(time, mean_knee, 'k-')
    ax3.plot(time, mean_ankle, 'k-')

    ax1.fill_between(time, smooth(mean_hip - std_hip, 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))

    ax1.set_ylabel("Nmm/KG", fontsize=20)
    ax2.set_ylabel("Nmm/KG", fontsize=20)
    ax3.set_ylabel("Nmm/KG", fontsize=20)
    ax1.set_title("Hip", fontsize=20)
    ax2.set_title("Knee", fontsize=20)
    ax3.set_title("Ankle", fontsize=20)
    plt.xlabel("Gait %", fontsize=20)

    plt.show()


def plot_leg_joints(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Angles', fontsize=20)
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




    print "Ankle: "
    print "Max Hip: ", np.max(np.abs(mean_hip)), " Std: ", std_hip[mean_hip.tolist().index(np.max(mean_hip))]
    print "Max Knee: ", np.max(np.abs(mean_knee)), " Std: ", std_knee[mean_knee.tolist().index(np.max(mean_knee))]
    print "Max Ankle: ", np.max(np.abs(mean_ankle)), " Std: ", std_ankle[mean_ankle.tolist().index(np.max(mean_ankle))]

    print "Min Hip: ", np.min(np.abs(mean_hip)), " Std: ", std_hip[mean_hip.tolist().index(np.min(mean_hip))]
    print "Min Knee: ", np.min(np.abs(mean_knee)), " Std: ", std_knee[mean_knee.tolist().index(np.min(mean_knee))]
    print "Min Ankle: ", np.min(np.abs(mean_ankle)), " Std: ", std_ankle[mean_ankle.tolist().index(np.min(mean_ankle))]

    ax1.plot(time, mean_hip, 'k-')
    ax2.plot(time, mean_knee, 'k-')
    ax3.plot(time, mean_ankle, 'k-')

    ax1.fill_between(time, smooth(mean_hip - std_hip, 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))

    ax1.set_ylabel("Degrees", fontsize=20)
    ax2.set_ylabel("Degrees", fontsize=20)
    ax3.set_ylabel("Degrees", fontsize=20)
    ax1.set_title("Hip", fontsize=20)
    ax2.set_title("Knee", fontsize=20)
    ax3.set_title("Ankle", fontsize=20)
    plt.xlabel("Gait %", fontsize=20)

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
    fig.suptitle('Walking data', fontsize=20)
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
    ax1.set_ylabel("Degrees", fontsize=20)
    ax2.set_ylabel("Nmm/Kg", fontsize=20)
    ax3.set_ylabel("W/Kg", fontsize=20)
    ax1.set_title("Angle", fontsize=20)
    ax2.set_title("Torque", fontsize=20)
    ax3.set_title("Power", fontsize=20)
    plt.xlabel("Gait %", fontsize=20)

    plt.show()

def plot_stairs(files):
    pass


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


if __name__ == "__main__":

    plot_stair_joint("/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_02/subject_02_stair_config1_03.csv")
    #
    # compare_stair_joints(["/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_00/subject_00 stairconfig1_01.csv",
    #                       "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_02/subject_02_stair_config1_03.csv"],
    #                      [(500, 800), (850, 1150) ])


    # compare_stair_joints(["/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_00/subject_00 stairconfig1_01.csv",
    #                       "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_02/subject_02_stair_config1_03.csv",
    #                       "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_03/subject_03_stair_config0_00.csv",
    #                       "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_04/subject_04_stair_config1_00.csv",
    #                       "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_05/subject_05_stair_config1_00.csv",
    #                       "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_06/subject_06 stairclimbing_config1_01.csv",
    #                       "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_07/subject_07 stairclimbing_config1_00.csv"],
    #                       [(500, 800),(850, 1200), (680, 980), (630, 920), (450, 720), (650, 980), (680, 1020)] )

    # plot_leg_joints(["/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_00/subject_00_walk_00.csv",
    #                  "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_01/subject_01_walk_00.csv",
    #                  "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_02/subject_02_walk_00.csv",
    #                  "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_03/subject_03_walk_00.csv",
    #                  "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_04/subject_04_walk_00.csv",
    #                  "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_05/subject_05_walk_00.csv",
    #                  "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_06/subject_06_walk_00.csv",
    #                  "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_07/subject_07_walk_00.csv"],
    #                  [1, 8, 16, 2, 11, 4, 4, 11],
    #                  ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7"])

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

    # plot_knee(["/home/nathaniel/Documents/MoCap_Participants/subject_07/subject_07_walk_00.csv"],
    #           [10],
    #           [ "Subject7"])
