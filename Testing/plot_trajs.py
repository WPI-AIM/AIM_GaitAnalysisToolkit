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
index = 0
ax.plot( np.arange(len(joints["LHipAngles"][index])) , np.radians(joints["LHipAngles"][index]) )
ax.plot( np.arange(len(joints["LKneeAngles"][index])) , np.radians(joints["LKneeAngles"][index]) )
ax.plot( np.arange(len(joints["LAbsAnkleAngle"][index])) , np.radians(joints["LAbsAnkleAngle"][index]) )

ax.plot( np.arange(len(joints["RHipAngles"][index])) , np.radians(joints["RHipAngles"][index]) )
ax.plot( np.arange(len(joints["RKneeAngles"][index])) , np.radians(joints["RKneeAngles"][index]) )
ax.plot( np.arange(len(joints["RAbsAnkleAngle"][index])) , np.radians(joints["RAbsAnkleAngle"][index]) )

ax.legend(["LHipAngles","LKneeAngles","LAbsAnkleAngle","RHipAngles", "RKneeAngles", "RAbsAnkleAngle"])
plt.show()
