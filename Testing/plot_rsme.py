from lib.Exoskeleton.Robot import core

from Vicon import Vicon, Markers
from Vicon import Markers
from Trajectories import rigid_marker
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt

hip_marker = [core.Point(0.0, 0.0, 0.0),
              core.Point(70.0, 0, 0.0),
              core.Point(0, 42.0, 0),
              core.Point(35.0, 70.0, 0.0)]


thigh_marker = [core.Point(0.0, 0.0, 0.0),
                core.Point(56.0, 0, 0.0),
                core.Point(0, 49.0, 0),
                core.Point(56.0, 63.0, 0.0)]

shank_marker = [core.Point(0.0, 0.0, 0.0),
                core.Point(56.0, 0, 0.0),
                core.Point(0, 42.0, 0),
                core.Point(56.0, 70.0, 0.0)]

data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort(True)
markers.auto_make_frames()

hip_err = []
thigh_err = []
shank_err = []

hip_frame = markers.get_rigid_body("ben:hip")
thigh_frame = markers.get_rigid_body("ben:RightThigh")
shank_frame = markers.get_rigid_body("ben:RightShank")

for frame in xrange(1000):
    m = markers.get_rigid_body("ben:hip")
    f = [ m[0][frame], m[1][frame], m[2][frame], m[3][frame] ]
    T, err = Markers.cloud_to_cloud(hip_marker, f)
    hip_err.append(err)

    m = markers.get_rigid_body("ben:RightShank")
    f = [ m[0][frame], m[1][frame], m[2][frame], m[3][frame] ]
    T, err = Markers.cloud_to_cloud(shank_marker, f)
    thigh_err.append(err)

    m = markers.get_rigid_body("ben:RightThigh")
    f = [ m[0][frame], m[1][frame], m[2][frame], m[3][frame] ]
    T, err = Markers.cloud_to_cloud(thigh_marker, f)
    shank_err.append(err)

plt.plot(hip_err)
plt.plot(thigh_err)
plt.plot(shank_err)
plt.xlabel("frame")
plt.ylabel("RMSE")
plt.show()