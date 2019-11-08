import matplotlib.pyplot as plt
import numpy as np
from EMG import EMG_Toolkit
from Session import Trial
import itertools
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

hip = []
knee = []
ankle = []
#joint = joints["LKneeAngles"]
for joint in joints["RHipAngles"]:
    hip.append( joint.data.tolist() )
hip = list(itertools.chain(*hip))

for joint in joints["RKneeAngles"]:
    knee.append( joint.data.tolist() )
knee = list(itertools.chain(*knee))
print joints.keys()
for joint in joints["LAbsAnkleAngle"]:
    ankle.append( joint.data.tolist())
ankle = list(itertools.chain(*ankle))


ax1.title.set_text('Hip Angle')
ax2.title.set_text('Knee Angle')
ax3.title.set_text('Ankle Angle')

ax1.set_ylabel("degs")
ax2.set_ylabel("degs")
ax3.set_xlabel("frames")
ax3.set_ylabel("deg")

ax1.plot(hip)
ax2.plot(knee)
ax3.plot(ankle)
plt.show()