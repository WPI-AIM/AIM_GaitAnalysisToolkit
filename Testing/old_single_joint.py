from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Vicon import Markers
from Utilities import Mean_filter
import time

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_autoscale_on(False)
data = Vicon.Vicon("/home/nathaniel/git/Gait_Analysis_Toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
cor_filter = Mean_filter.Mean_Filter(1)
aor_filter = Mean_filter.Mean_Filter(1)
all_core = []


def animate(frame):
    frame = frame
    x = []
    y = []
    z = []

    m = markers.get_rigid_body("ben:hip")[0:]
    x += [m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x]
    y += [m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y]
    z += [m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    m = markers.get_rigid_body("ben:RightThigh")[0:]
    x += [m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x]
    y += [m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y]
    z += [m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    m = markers.get_rigid_body("ben:RightShank")[0:]
    x += [m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x]
    y += [m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y]
    z += [m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    ax.clear()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis([-500, 500, -750, 1500])
    ax.set_zlim3d(0, 1250)
    ax.scatter(x, y, z, c='r', marker='o')
    offset = 3
    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:]

    shank_markers = markers.get_rigid_body("ben:RightShank")[0:1000]

    m1 = shank_markers[0][frame:frame + 2]
    m2 = shank_markers[1][frame:frame + 2]
    m3 = shank_markers[2][frame:frame + 2]
    m4 = shank_markers[3][frame:frame + 2]
    data = [m1, m2, m3, m4]
    core = cor_filter.update(Markers.calc_CoR(data))
    axis = aor_filter.update(Markers.calc_AoR(data))
    axis_x = [(core[0] - axis[0] * 1000).item(0), (core[0]).item(0), (core[0] + axis[0] * 1000).item(0)]
    axis_y = [(core[1] - axis[1] * 1000).item(0), (core[1]).item(0), (core[1] + axis[1] * 1000).item(0)]
    axis_z = [(core[2] - axis[2] * 1000).item(0), core[2].item(0), (core[2] + axis[2] * 1000).item(0)]
    # axis_x = [core[0]]
    # axis_y = [core[1]]
    # axis_z = [core[2]]
    ax.scatter( core[0], core[1], core[2], 'go')
    ax.plot(axis_x, axis_y, axis_z, 'b')

    #ax.scatter(Rc[0], Rc[1], Rc[2], 'b')


ani = animation.FuncAnimation(fig, animate, interval=100)

plt.show()
