import matplotlib.pyplot as plt

from EMG import EMG_Toolkit
from Session import Trial

vicon_file = "/media/nathaniel/Data/LowerLimb_HealthyGait/Subject02/walking/Walking01.csv"
config_file = "/home/nathaniel/git/exoserver/Config/sensor_list.yaml"
exo_file = "/home/nathaniel/git/exoserver/Main/subject_1234_trial_1.csv"
trial = Trial.Trial(vicon_file, config_file, exo_file)
joints = trial.get_joint_trajectories()
plate = trial.get_force_plates()
cop = trial.get_CoP()
emgs = trial.get_emg()

fig, (ax1, ax2, ax3) = plt.subplots(3)
#joint = joints["LKneeAngles"]
for joint in joints["RKneeAngles"]:
    ax1.plot(joint.time, joint.data )

print "keys", emgs.keys()
for emg in emgs[1]:
    data = EMG_Toolkit.remove_mean(emg.data)
    data = EMG_Toolkit.butterworth(data)
    data = EMG_Toolkit.rectify(data)
    data = EMG_Toolkit.low_pass(data)
    ax2.plot(data)
for emg in emgs[7]:
    data = EMG_Toolkit.remove_mean(emg.data)
    data = EMG_Toolkit.butterworth(data)
    data = EMG_Toolkit.rectify(data)
    data = EMG_Toolkit.low_pass(data)
    ax3.plot(data)

plt.ylabel('some numbers')
plt.show()
