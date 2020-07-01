import sys
import matplotlib.pyplot as plt
import Core.utilities as utilities
from scipy import signal
import numpy as np
from Session import ViconGaitingTrial
import Vicon
import os
file = "/home/nathanielgoldfarb/Downloads/Subject01_ver03.csv"
trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
joints = trial.get_joint_trajectories()
leg = trial.vicon.get_model_output().get_left_leg()
plt.plot(leg.knee.angle.x)
plt.show()
