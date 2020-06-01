import sys
import matplotlib.pyplot as plt
from scipy import signal
import numpy as np
from lib.GaitAnalysisToolkit.Session import ViconGaitingTrial
from lib.GaitAnalysisToolkit.lib.Vicon import Vicon


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
    z = smooth(map(int, z), 5)
    z = np.convolve(z, np.ones((N,)) / N, mode='valid')

    max_peakind = np.diff(np.sign(np.diff(z))).flatten()  # the one liner
    max_peakind = np.pad(max_peakind, (1, 1), 'constant', constant_values=(0, 0))
    max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]
    secound_step = max_peakind[-1]
    first_step = max_peakind[-2]

    index = secound_step
    while z[index] != z[index + 1]:
        print index
        index += 1
    final_index = index

    index = first_step
    while z[index] != z[index - 1]:
        index -= 1
    start_index = index
    # plt.plot(z)
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
    font_size = 25
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
        hip.append(signal.resample(joints["Rhip"][i].angle.data, resample))
        knee.append(signal.resample(joints["Rknee"][i].angle.data, resample))
        ankle.append(signal.resample(joints["Rankle"][i].angle.data, resample))

    time = np.linspace(0, 1, resample)
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
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        print "file ", file
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip.append(signal.resample(abs(joints["Rhip"][i].moment.data), resample))
        knee.append(signal.resample(abs(joints["Rknee"][i].moment.data), resample))
        ankle.append(signal.resample(abs(joints["Rankle"][i].moment.data), resample))

    time = np.linspace(0, 1, resample)
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
        hip.append(signal.resample(abs(joints["Rhip"][i].power.data), resample))
        knee.append(signal.resample(abs(joints["Rknee"][i].power.data), resample))
        ankle.append(signal.resample(abs(joints["Rankle"][i].power.data), resample))

    time = np.linspace(0, 1, resample)
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
        hip.append(abs(signal.resample(joints["Rhip"][i].moment.data), resample))
        knee.append(abs(signal.resample(joints["Rknee"][i].moment.data), resample))
        ankle.append(abs(signal.resample(joints["Rankle"][i].moment.data), resample))

    time = np.linspace(0, 1, resample)
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


def plot_signle_knee(file):
    trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
    joints = trial.get_joint_trajectories()
    leg = []
    for i in xrange(len(joints["Rknee"])):
        leg.append(i)
        plt.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].angle.data)
    plt.legend(leg)
    plt.show()


def plot_knee(files, list_of_index):
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['xtick.labelsize'] = 25
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Knee Joint', fontsize=20)
    angle = []
    power = []
    moment = []
    time = None
    resample = 100000
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        trial.create_index_seperators()
        joints = trial.get_joint_trajectories()
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        trial.create_index_seperators()
        joints = trial.get_joint_trajectories()
        angle.append(signal.resample(joints["Rknee"][i].angle.data, resample))
        power.append(signal.resample(joints["Rknee"][i].power.data, resample))
        moment.append(signal.resample(joints["Rknee"][i].moment.data, resample))

    time = np.linspace(0, 1, resample)
    angle = np.array(angle)
    power = np.array(power)
    moment = np.array(moment)

    mean_angle = smooth(np.mean(angle, axis=0), 5)
    mean_power = smooth(np.mean(power, axis=0), 5)
    mean_moment = smooth(np.mean(moment, axis=0), 5)

    std_hip = np.std(angle, axis=0)
    std_knee = np.std(power, axis=0)
    std_ankle = np.std(moment, axis=0)



    ax1.plot(time, mean_angle, 'k-', linewidth=4)
    ax2.plot(time, mean_power, 'k-', linewidth=4)
    ax3.plot(time, mean_moment, 'k-', linewidth=4)

    ax1.fill_between(time, smooth(mean_angle - std_hip, 5), smooth(mean_angle + std_hip, 5))
    ax2.fill_between(time, smooth(mean_power - std_knee, 5), smooth(mean_power + std_knee, 5))
    ax3.fill_between(time, smooth(mean_moment - std_ankle, 5), smooth(mean_moment + std_ankle, 5))

    ax1.set_ylabel("Degrees", fontsize=30)
    ax2.set_ylabel("W/Kg", fontsize=30)
    ax3.set_ylabel("Nmm/Kg", fontsize=30)
    ax1.set_title("Angle", fontsize=30)
    ax2.set_title("Power", fontsize=30)
    ax3.set_title("Moment", fontsize=20)
    plt.xlabel("Gait %", fontsize=20)

    plt.show()
#
#
# def plot_knee_stairs(files, side):
#     fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
#     fig.suptitle('Stair Knee Joint', fontsize=20)
#     angle = []
#     power = []
#     moment = []
#     time = None
#     resample = 100000
#
#     indiecs = {}
#     for file, s in zip(files, side):
#         rn = get_stair_ranges(file, s)
#         indiecs[file] = rn
#         resample = min(resample, rn[1] - rn[0])
#
#     for file, s in zip(files, side):
#         trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
#         if s == "R":
#             joints = trial.vicon.get_model_output().get_right_leg()
#         else:
#             joints = trial.vicon.get_model_output().get_left_leg()
#         rn = indiecs[file]
#         angle.append(signal.resample(joints.knee.angle.x[rn[0]:rn[1]], resample))
#         power.append(signal.resample(joints.knee.power.z[rn[0]:rn[1]], resample))
#         moment.append(signal.resample(joints.knee.moment.x[rn[0]:rn[1]], resample))
#
#     time = np.linspace(0, 1, resample)
#     angle = np.array(angle)
#     power = np.array(power)
#     moment = np.array(moment)
#
#     mean_angle = smooth(np.mean(angle, axis=0), 5)
#     mean_power = smooth(np.mean(power, axis=0), 5)
#     mean_moment = smooth(np.mean(moment, axis=0), 5)
#
#     std_hip = np.std(angle, axis=0)
#     std_knee = np.std(power, axis=0)
#     std_ankle = np.std(moment, axis=0)
#
#     ax1.plot(time, mean_angle, 'k-', linewidth=4)
#     ax2.plot(time, mean_power, 'k-', linewidth=4)
#     ax3.plot(time, mean_moment, 'k-', linewidth=4)
#
#     ax1.fill_between(time, smooth(mean_angle - std_hip, 5), smooth(mean_angle + std_hip, 5))
#     ax2.fill_between(time, smooth(mean_power - std_knee, 5), smooth(mean_power + std_knee, 5))
#     ax3.fill_between(time, smooth(mean_moment - std_ankle, 5), smooth(mean_moment + std_ankle, 5))
#
#     ax1.set_ylabel("Degrees", fontsize=20)
#     ax2.set_ylabel("W/Kg", fontsize=20)
#     ax3.set_ylabel("Nmm/Kg", fontsize=20)
#     ax1.set_title("Angle", fontsize=20)
#     ax2.set_title("Power", fontsize=20)
#     ax3.set_title("Moment", fontsize=20)
#     plt.xlabel("Gait %", fontsize=20)
#
#     plt.show()


def smooth(y, box_pts):
    box = np.ones(box_pts) / box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth


def plot_joint(file, index):
    trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
    trial.create_index_seperators()
    joints = trial.get_joint_trajectories()
    leg = []
    plt.plot(joints["Rknee"][index].angle.time, joints["Rknee"][index].angle.data)
    plt.show()


def sit_to_stand(file):
    trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
    joints = trial.vicon.get_model_output().get_left_leg()

    angles = joints.knee.angle.x
    power = joints.knee.power.z
    moment = joints.knee.moment.x
    time = np.linspace(0, 1, len(moment))
    plt.rcParams.update({'font.size': 22})
    plt.rcParams['xtick.labelsize'] = 25
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Stair Knee Joint', fontsize=30)

    ax1.plot(time, angles, linewidth=4)
    ax2.plot(time, power, linewidth=4)
    ax3.plot(time, moment, linewidth=4)

    ax1.set_ylabel("Degrees", fontsize=30)
    ax2.set_ylabel("W/Kg", fontsize=30)
    ax3.set_ylabel("Nmm/Kg", fontsize=30)
    ax1.set_title("Angle", fontsize=30)
    ax2.set_title("Power", fontsize=30)
    ax3.set_title("Moment", fontsize=30)
    plt.xlabel("Gait %", fontsize=30)

    plt.show()


if __name__ == "__main__":
    #
    # compare_stair_angles(
    #     ["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 stairconfig1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_stair_config1_03.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_stair_config0_02.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_stair_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_stair_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 stairclimbing_config1_02.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 stairclimbing_config1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_stair_config1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_09/subject_09 stairclimbing_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 stairclimbing_config1_00.csv"],
    #     ["R", "R", "L", "L", "R", "L", "R", "L", "R", "R"],
    #     ["subject00", "subject02", "Subject03", "Subject04", "Subject05", "Subject06", "Subject07", "Subject08",
    #      "Subject09", "Subject10"])
    #
    # compare_stair_power(
    #     ["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 stairconfig1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_stair_config1_03.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_stair_config0_02.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_stair_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_stair_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 stairclimbing_config1_02.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 stairclimbing_config1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_stair_config1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_09/subject_09 stairclimbing_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 stairclimbing_config1_00.csv"],
    #     ["R", "R", "L", "L", "R", "L", "R", "L", "R", "R"],
    #     ["subject00", "subject02", "Subject03", "Subject04", "Subject05", "Subject06", "Subject07", "Subject08",
    #      "Subject09", "Subject10"])
    #
    # compare_stair_moments(
    #     ["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 stairconfig1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_stair_config1_03.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_stair_config0_02.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_stair_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_stair_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 stairclimbing_config1_02.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 stairclimbing_config1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_stair_config1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_09/subject_09 stairclimbing_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 stairclimbing_config1_00.csv"],
    #     ["R", "R", "L", "L", "R", "L", "R", "L", "R", "R"],
    #     ["subject00", "subject02", "Subject03", "Subject04", "Subject05", "Subject06", "Subject07", "Subject08",
    #      "Subject09", "Subject10"])

    # compare_walking_moments(
    #     ["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_01/subject_01_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_walking_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 walk_00.csv"],
    #     [1, 8, 16, 2, 11, 4, 4, 11, 16, 9, 9],
    #     ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7", "Subject8",
    #      "Subject10"])
    #
    # compare_walking_power(
    #     ["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_01/subject_01_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_walking_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 walk_00.csv"],
    #     [1, 8, 16, 2, 11, 4, 4, 11, 16, 9, 9],
    #     ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7", "Subject8",
    #      "Subject10"])

    compare_walking_angles(
        ["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_01/subject_01_walk_00.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_walk_00.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_walk_00.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_walk_00.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_walk_00.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 walk_00.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 walk_00.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_walking_01.csv",
         "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 walk_00.csv"],
        [1, 8, 16, 2, 11, 4, 4, 11, 16, 9, 9],
        ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7", "Subject8",
         "Subject10"])

    # compare_walking_power(
    #     ["/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_00/subject_00_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_01/subject_01_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_02/subject_02_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_03/subject_03_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_04/subject_04_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_05/subject_05_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_06/subject_06_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_07/subject_07_walk_00.csv"],
    #     [1, 8, 16, 2, 11, 4, 4, 11],
    #     ["Subject0", "Subject1", "Subject2", "Subject3", "Subject4", "Subject5", "Subject6", "Subject7"])

    # compare_stair_power(
    #     ["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 stairconfig1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_stair_config1_03.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_stair_config0_02.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_stair_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_stair_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 stairclimbing_config1_02.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 stairclimbing_config1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_stair_config1_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_09/subject_09 stairclimbing_config1_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 stairclimbing_config1_00.csv"],
    #     ["R", "R", "L", "L", "R", "L", "R", "L", "R", "R"],
    #     ["subject00", "subject02", "Subject03", "Subject04", "Subject05", "Subject06", "Subject07", "Subject08",
    #      "Subject09", "Subject10"])

    #plot_joint("/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv", 1 )
    #plot_joint("/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_01/subject_01_walk_00.csv", 7)
    #plot_joint( "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_walk_01.csv", 11)
    #plot_joint("/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_walk_00.csv", 0)
    #plot_joint("/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_walk_00.csv", 12)
    #plot_joint("/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_walk_00.csv",4)
    #plot_joint("/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 walk_00.csv",3)
    #plot_joint("/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 walk_00.csv", 6)
    #plot_joint("/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 walk_00.csv", 8)


    # plot_knee(["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv",
    #            "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_01/subject_01_walk_00.csv",
    #            "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_walk_01.csv",
    #            "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_walk_00.csv",
    #            "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_walk_00.csv",
    #            "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_walk_00.csv",
    #            "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 walk_00.csv",
    #            "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 walk_00.csv",
    #            "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 walk_00.csv"],
    #           [1,7,11,0,12,4,3,6,8])


    # plot_knee(
    #     ["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_01/subject_01_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 walk_00.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_walking_01.csv",
    #      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 walk_00.csv"],
    #     [1, 7, 16, 2, 11, 4, 4, 11, 16, 9, 9])

    #
    #
    # plot_knee_stairs(["/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_00/subject_00 stairconfig1_01.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_02/subject_02_stair_config1_03.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_03/subject_03_stair_config0_02.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_04/subject_04_stair_config1_00.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_05/subject_05_stair_config1_00.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_06/subject_06 stairclimbing_config1_02.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_07/subject_07 stairclimbing_config1_01.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_08/subject_08_stair_config1_01.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_09/subject_09 stairclimbing_config1_00.csv",
    #                      "/home/nathanielgoldfarb/Documents/stairclimbing_data/CSVs/subject_10/subject_10 stairclimbing_config1_00.csv"],
    #                     ["R", "R", "L", "L", "R", "L", "R", "L", "R", "R"] )


    # sit_to_stand("/media/nathanielgoldfarb/My Book/BioMechanocle_knee/Subject00/12162019/Subject00_Sit_to_stand_00.csv")