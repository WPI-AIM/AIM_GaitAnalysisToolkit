
import matplotlib.pyplot as plt
import numpy as np
from lib.dmp_experiments.Python import train_dmp, DMP_runner
from Session import Trial
from Vicon import Vicon

trial = Trial.Trial(vicon_file="/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_00/11_07_2019/subject_00 stairconfig1_03.csv")
traj = trial.vicon.get_model_output().get_right_leg().hip.angle.x[538:750]

T = Trial.calc_kinematics(traj)
#
#Name of the file
name = 'Simple_dmps.xml'

#Set no. of basis functions
n_rfs = 200

#Set the time-step
dt = 0.01 #traj.time[1] - traj.time[0]

Important_values = train_dmp.train_dmp(name, n_rfs, T, dt)
start = traj[0]
goal = traj[-1]
my_runner = DMP_runner.DMP_runner(name, start, goal)

Y = []
tau = 1.0

for i in np.arange(0, int(tau/dt + 1.0)):
    my_runner.step(tau,dt)
    Y.append(my_runner.y)

#my_runner = DMP_runner(name,start,goal)

#
# plt.xlabel("Frame")
# plt.ylabel("Angle")
#
plt.plot(traj)
plt.plot(Y)
# plt.legend(["raw", "DMP"])
plt.show()
