import matplotlib.pyplot as plt
import numpy as np
from EMG import EMG_Toolkit
from Session import Trial

vicon_file = "/home/nathaniel/Downloads/Walking01.csv"
config_file = "/home/nathaniel/exoserver/Config/sensor_list.yaml"
exo_file = "/home/nathaniel/exoserver/Main/subject_1234_trial_1.csv"
trial = Trial.Trial(vicon_file, config_file, exo_file)
joints = trial.get_joint_trajectories()
plate = trial.get_force_plates()
#cop = trial.get_CoP()
emgs = trial.get_emg()
#trial.plot()

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

#joint = joints["LKneeAngles"]
for joint in joints["RKneeAngles"]:
    ax1.plot( 100.0 * joint.time/max(joint.time), joint.data )

for emg in emgs[1]:
    data = EMG_Toolkit.remove_mean(emg.data)
    data = EMG_Toolkit.butterworth(data)
    data = EMG_Toolkit.rectify(data)
    data = EMG_Toolkit.low_pass(data)
    time = np.linspace(0, 100, len(data))
    ax2.plot(time, data)
for emg in emgs[7]:
    data = EMG_Toolkit.remove_mean(emg.data)
    data = EMG_Toolkit.butterworth(data)
    data = EMG_Toolkit.rectify(data)
    data = EMG_Toolkit.low_pass(data)
    time = np.linspace(0,100,len(data))
    ax3.plot(time, data)

ax1.title.set_text('Knee Angle')
ax2.title.set_text('EMG Thigh')
ax3.title.set_text('EMG Shank')

ax1.set_ylabel("degs")

ax2.set_ylabel("mv")
ax3.set_xlabel("% of gait cycle")
ax3.set_ylabel("mV")
plt.show()
