from Trajectories import trajecectory_helper
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from lib.dmp_experiments.Python import train_rmp, RMP_runner
from math import radians

file = "/home/nathaniel/git/AMBF_Walker/config/joint_data_edited.csv"
name = "LHipAngles"

joints = trajecectory_helper.sperate_joints(file)
#trajecectory_helper.sperate_joints2(file)
fig, ax = plt.subplots()
fig2, ax2 = plt.subplots()

data = train_rmp.train_rmp("ankle.xml", 1000, np.radians(np.array([joints["LAbsAnkleAngle"][2]])) , 0.01)
data = train_rmp.train_rmp("knee.xml", 1000, np.radians(np.array([joints["LKneeAngles"][2]])) , 0.01)
data = train_rmp.train_rmp("hip.xml", 1000, np.radians(np.array([joints["LHipAngles"][2]])) , 0.01)
ax.plot( np.arange(len(joints["LKneeAngles"][2])) , np.radians(joints["LKneeAngles"][2]) )

runner = RMP_runner.RMP_runner("knee.xml")

y,dy,ddy = runner.run()
ax2.plot( np.arange(len(y) ), y )
print
#print y[0] -  np.radians(joints["LKneeAngles"][2][0])
plt.show()
