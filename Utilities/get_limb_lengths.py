from Vicon import Vicon
import numpy as np
import matplotlib.pyplot as plt
from Trajectories import center_of_rotation



data = Vicon.Vicon("/home/nathanielgoldfarb/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
markers.play()

