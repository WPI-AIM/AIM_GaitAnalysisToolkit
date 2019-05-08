
import pandas as pd
from Trajectories import trajecectory_helper
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lib.dmp_experiments.Python import train_rmp, RMP_runner
from math import radians
import scipy.interpolate

file = "/home/nathaniel/git/control/Control/Tasks/Walk/knee_traj.csv"
name = "LHipAngles"
gait_data = pd.read_csv(file)

data = np.radians(np.array(gait_data).flatten())

dt = .01
timesteps = int(1. / dt)
path = np.zeros(timesteps)
x = np.linspace(0, 1, len(data))
path_gen = scipy.interpolate.interp1d(x, data)
for t in range(timesteps):
    path[t] = path_gen(t * dt)
path1 = np.cos(np.arange(0, 2 * np.pi, .01) * 5)
T = np.array([path1])
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()
# #
train_rmp.train_rmp("wolf3.xml", 1000, np.array([path]) , 0.01)
# fig, ax = plt.subplots()
ax.plot( np.arange(len(path)) , path )
#
runner = RMP_runner.RMP_runner("wolf3.xml")
# # print np.radians(joints["LHipAngles"][1][0])
# # print np.radians(joints["LHipAngles"][1][-1])
y, dy, ddy = runner.run()
ax2.plot( np.arange(len(y) ), y )
# # print
# # #print y[0] -  np.radians(joints["LKneeAngles"][2][0])
plt.show()
