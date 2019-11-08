from Vicon import Vicon
import numpy as np
import matplotlib.pyplot as plt
from Trajectories import center_of_rotation
from lib.Exoskeleton.Robot import core
from scipy import signal

file = "/home/nathanielgoldfarb/gait_analysis_toolkit/testing_data/stairclimb03.csv"
data = Vicon.Vicon(file)
markers = data.get_markers()
markers.smart_sort()

markers.play()