import sys
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from Session import ViconGaitingTrial
from lib.Vicon import Vicon
import os


def plot_stair_joint(file):

    trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
    joints = trial.vicon.get_model_output().get_right_leg()
    plt.plot(joints.hip.angle.x)
    plt.plot(joints.knee.angle.x)
    plt.plot(joints.ankle.angle.x)
    print get_stair_ranges(file)
    plt.legend(["x", "y", "z"])
    plt.show()


def get_stair_ranges(file, side="R"):
    print file
    trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
    if side == "R":
        m = trial.vicon.markers.get_marker("RTOE")
    else:
        m = trial.vicon.markers.get_marker("LTOE")

    z = []
    for i in xrange(len(m)):
        z.append(m[i].z)

    N = 10
    z = smooth(map(int, z) ,5)
    z = np.convolve(z, np.ones((N,)) / N, mode='valid')

    max_peakind = np.diff(np.sign(np.diff(z))).flatten()  # the one liner
    max_peakind = np.pad(max_peakind, (1, 1), 'constant', constant_values=(0, 0))
    max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]
    secound_step = max_peakind[-1]
    first_step = max_peakind[-2]

    index = secound_step
    while z[index] != z[index+1]:
        print index
        index += 1
    final_index = index

    index = first_step
    while z[index] != z[index - 1]:
        index -= 1
    start_index = index
    #plt.plot(z)
    return (start_index, final_index)


def compare_stair_angles(files, side, legend):

    resample = 100000
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Stair Joint Angles', fontsize=20)
    
    hip = []
    knee = []
    ankle = []

    resample = 1000000000

    indiecs = {}
    for file, s in zip(files, side):
        rn = get_stair_ranges(file,s)
        indiecs[file] = rn
        resample = min(resample, rn[1] - rn[0])

    for file, s in zip(files,side):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        if s == "R":
            joints = trial.vicon.get_model_output().get_right_leg()
        else:
            joints = trial.vicon.get_model_output().get_left_leg()
        rn = indiecs[file]
        hip.append(signal.resample(joints.hip.angle.x[rn[0]:rn[1]], resample))
        knee.append(signal.resample(joints.knee.angle.x[rn[0]:rn[1]], resample))
        ankle.append(signal.resample(joints.ankle.angle.x[rn[0]:rn[1]], resample))

    mean_hip = smooth(np.mean(hip, axis=0), 5)
    mean_knee = smooth(np.mean(knee, axis=0), 5)
    mean_ankle = smooth(np.mean(ankle, axis=0), 5)

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)
    time = np.linspace(0, 1, resample)
    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)


    ax1.fill_between(time, smooth(mean_hip - std_hip, 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))
    font_size=25
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


    hip = []
    knee = []
    ankle = []

    resample = 1000000000

    indiecs = {}
    for file, s in zip(files, side):
        rn = get_stair_ranges(file, s)
        indiecs[file] = rn
        resample = min(resample, rn[1] - rn[0])

    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        if s == "R":
            joints = trial.vicon.get_model_output().get_right_leg()
        else:
            joints = trial.vicon.get_model_output().get_left_leg()
        rn = indiecs[file]
        hip.append(signal.resample(np.abs(joints.hip.moment.x[rn[0]:rn[1]]), resample))
        knee.append(signal.resample(np.abs(joints.knee.moment.x[rn[0]:rn[1]]), resample))
        ankle.append(signal.resample(np.abs(joints.ankle.moment.x[rn[0]:rn[1]]), resample))

    mean_hip = smooth(np.mean(hip, axis=0), 5)
    mean_knee = smooth(np.mean(knee, axis=0), 5)
    mean_ankle = smooth(np.mean(ankle, axis=0), 5)

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)
    time = np.linspace(0, 1, resample)
    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)

    ax1.fill_between(time, smooth(mean_hip - std_hip, 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))
    font_size = 25
    ax1.set_ylabel("Nmm/Kg", fontsize=font_size),
    ax2.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax3.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()


def compare_stair_power(files, side, legend):
    print "asldjflasdjf"
    resample = 100000
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Stair Joint Power', fontsize=20)

    hip = []
    knee = []
    ankle = []

    resample = 1000000000

    indiecs = {}
    for file, s in zip(files, side):
        rn = get_stair_ranges(file, s)
        indiecs[file] = rn
        resample = min(resample, rn[1] - rn[0])

    for file, s in zip(files, side):

        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        if s == "R":
            joints = trial.vicon.get_model_output().get_right_leg()
        else:
            joints = trial.vicon.get_model_output().get_left_leg()
        rn = indiecs[file]
        hip.append(signal.resample(np.abs(joints.hip.power.z[rn[0]:rn[1]]), resample))
        knee.append(signal.resample(np.abs(joints.knee.power.z[rn[0]:rn[1]]), resample))
        ankle.append(signal.resample(np.abs(joints.ankle.power.z[rn[0]:rn[1]]), resample))

    mean_hip = smooth(np.mean(hip, axis=0), 5)
    mean_knee = smooth(np.mean(knee, axis=0), 5)
    mean_ankle = smooth(np.mean(ankle, axis=0), 5)

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)
    time = np.linspace(0, 1, resample)
    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)

    ax1.fill_between(time, smooth(mean_hip - std_hip, 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))
    font_size=25
    ax1.set_ylabel("W/Kg", fontsize=font_size),
    ax2.set_ylabel("W/Kg", fontsize=font_size)
    ax3.set_ylabel("W/Kg", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()


def compare_walking_angles(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Angles', fontsize=20)
    hip = []
    knee = []
    ankle = []
    time = None
    resample = 100000
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):

        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
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
    mean_ankle = smooth(np.mean(ankle, axis=0), 5)

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

    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)


    ax1.fill_between(time, smooth(mean_hip - std_hip, 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))
    font_size = 25
    ax1.set_ylabel("Degrees", fontsize=font_size)
    ax2.set_ylabel("Degrees", fontsize=font_size)
    ax3.set_ylabel("Degrees", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()


def compare_walking_moments(files, list_of_index, legend=None):


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

    mean_hip = smooth(np.mean(hip, axis=0), 5)
    mean_knee = smooth(np.mean(knee, axis=0), 5)
    mean_ankle = smooth(np.mean(ankle, axis=0), 5   )

    std_hip = np.std(hip, axis=0)
    std_knee = np.std(knee, axis=0)
    std_ankle = np.std(ankle, axis=0)


    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)


    ax1.fill_between(time, smooth(mean_hip - std_hip, 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))
    font_size = 25
    ax1.set_ylabel("Nmm/KG", fontsize=font_size)
    ax2.set_ylabel("Nmm/KG", fontsize=font_size)
    ax3.set_ylabel("Nmm/KG", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()

def compare_walking_power(files, list_of_index, legend=None):


    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Power', fontsize=20)
    hip = []
    knee = []
    ankle = []
    time = None
    resample = 100000
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        print "file ", file
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):

        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip.append( signal.resample( abs(joints["Rhip"][i].power.data), resample))
        knee.append(signal.resample( abs(joints["Rknee"][i].power.data), resample))
        ankle.append(signal.resample( abs(joints["Rankle"][i].power.data), resample))


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


    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)


    ax1.fill_between(time, smooth(mean_hip - std_hip , 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))

    ax1.set_ylabel("W/Kg", fontsize=20)
    ax2.set_ylabel("W/Kg", fontsize=20)
    ax3.set_ylabel("W/Kg", fontsize=20)
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
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):

        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip.append(abs(signal.resample( joints["Rhip"][i].moment.data), resample))
        knee.append(abs(signal.resample( joints["Rknee"][i].moment.data), resample))
        ankle.append(abs(signal.resample( joints["Rankle"][i].moment.data), resample))


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


    ax1.plot(time, mean_hip, 'k-', linewidth=4)
    ax2.plot(time, mean_knee, 'k-', linewidth=4)
    ax3.plot(time, mean_ankle, 'k-', linewidth=4)

    ax1.fill_between(time, smooth(mean_hip - std_hip, 5), smooth(mean_hip + std_hip, 5))
    ax2.fill_between(time, smooth(mean_knee - std_knee, 5), smooth(mean_knee + std_knee, 5))
    ax3.fill_between(time, smooth(mean_ankle - std_ankle, 5), smooth(mean_ankle + std_ankle, 5))

    ax1.set_ylabel("Nmm/Kg", fontsize=20)
    ax2.set_ylabel("Nmm/Kg", fontsize=20)
    ax3.set_ylabel("Nmm/Kg", fontsize=20)
    ax1.set_title("Hip", fontsize=20)
    ax2.set_title("Knee", fontsize=20)
    ax3.set_title("Ankle", fontsize=20)
    plt.xlabel("Gait %", fontsize=20)

    plt.show()

def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


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
        [1, 8],
        ["Subject0", "Subject1"])

    compare_walking_moments(
        [os.path.join(script_dir, "ExampleData/subject_00 walk_00.csv"),
         os.path.join(script_dir, "ExampleData/subject_01 walk_00.csv")],
        [1, 8],
        ["Subject0", "Subject1"])
