from Trajectories import sperate_trajectories
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
file = "/home/nathaniel/git/AMBF_Walker/config/joint_data.csv"
name = "LHipAngles"

joints = sperate_trajectories.sperate_joints(file)
fig, ax = plt.subplots()
ax.plot(np.arange(len(joints["LHipAngles"][0])), joints["LHipAngles"][0])
ax.plot(np.arange(len(joints["RHipAngles"][0])), joints["RHipAngles"][0])
fig, ax = plt.subplots()
plt.show()