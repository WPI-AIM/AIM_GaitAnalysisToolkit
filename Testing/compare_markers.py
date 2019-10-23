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

body = markers.get_rigid_body("ben:RightShank")

marker = 2
x = []
y = []
z = []

for frame in xrange(1000):
    m = body[marker][frame]
    x.append(m.x)
    y.append(m.y)
    z.append(m.z)

plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.xlabel("frame")
plt.ylabel("RMSE")
plt.show()