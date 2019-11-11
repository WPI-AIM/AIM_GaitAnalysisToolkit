import matplotlib.pyplot as plt
import numpy as np
from lib.dmp_experiments.Python import train_dmp, DMP_runner
from Session import Trial
from Vicon import Vicon
#data = Vicon.Vicon("/home/nathanielgoldfarb/gait_analysis_toolkit/testing_data/Range_of_Motion.csv")
#data = Vicon.Vicon("/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_00/11_07_2019/subject_00 stairconfig4_03.csv")
trial = Trial.Trial(vicon_file="/home/nathanielgoldfarb/stairclimbing/CSV_files/subject_00/11_07_2019/subject_00 walk_01.csv")
#joint = trial.vicon.get_model_output().get_right_leg().hip.angle.x
traj = trial.get_joint_trajectories()["Rhip"][0]
T = Trial.calc_kinematics(traj)
#Name of the file
name = 'Simple_dmps.xml'

#Set no. of basis functions
n_rfs = 200

#Set the time-step
dt = traj.time[-1]

Important_values =  train_dmp.train_dmp(name, n_rfs, T, dt)
start = traj.data[0]
goal = traj.data[-1]
my_runner = DMP_runner.DMP_runner(name,start,goal)

Y = []
tau = 1
Y = []
print tau/dt
for i in np.arange(0,int(tau/dt)+1):

    '''Dynamic change in goal'''
    #new_goal = 2
    #new_flag = 1
    #if i > 0.6*int(tau/dt):
    #    my_runner.setGoal(new_goal,new_flag)
    '''Dynamic change in goal'''

    my_runner.step(tau,dt)
    Y.append(my_runner.y)
#my_runner = DMP_runner(name,start,goal)
print Y
plt.xlabel("Frame")
plt.ylabel("Angle")
#plt.plot(traj.time, traj.data)
plt.plot(Y)
plt.show()
