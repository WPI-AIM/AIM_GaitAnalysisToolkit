from Vicon import Vicon
from lib.Exoskeleton.Robot import core
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
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
all_core = []
angles = []
shank = markers.get_rigid_body("ben:RightShank")
thigh = markers.get_rigid_body("ben:RightThigh")
shank_body = markers.get_rigid_body("ben:RightShank")[0:]
thigh_body = markers.get_rigid_body("ben:RightThigh")[0:]

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



T_hip = []
T_thigh = []
T_shank = []

for frame in xrange(1000):
    m = markers.get_rigid_body("ben:hip")
    f = [m[0][frame], m[1][frame], m[2][frame],m[3][frame] ]
    T, err = Markers.cloud_to_cloud(hip_marker, f)

    T_hip.append(T)

    m = markers.get_rigid_body("ben:RightShank")
    f = [ m[0][frame], m[1][frame], m[2][frame], m[3][frame] ]
    T, err = Markers.cloud_to_cloud(shank_marker, f)

    T_shank.append(T)

    m = markers.get_rigid_body("ben:RightThigh")
    f = [ m[0][frame], m[1][frame], m[2][frame], m[3][frame] ]
    T, err = Markers.cloud_to_cloud(thigh_marker, f)

    T_thigh.append(T)

def animate(frame):

    x = []
    y = []
    z = []

    T1 = np.dot(np.linalg.pinv(T_thigh[frame]), T_shank[frame])
    T2 = np.dot(np.linalg.pinv(T_thigh[frame+1]), T_shank[frame+1])
    m = markers.get_rigid_body("ben:RightShank")

    f1 = [m[0][frame], m[1][frame], m[2][frame], m[3][frame]]
    f2 = [m[0][frame+5], m[1][frame+5], m[2][frame+5], m[3][frame+5]]

    T, err = Markers.cloud_to_cloud(f1, f2)
    xc = Markers.get_center([f1,f2], T[:3,:3])
    xc = xc +  T[3,0:3].reshape((-1,1))
    xc = np.append(xc, [[1.0]], axis=0)
    xc = np.dot(T_shank[frame], xc)
    axis, angle = Markers.R_to_axis_angle(T[0:3, 0:3])

    #sol = Markers.minimize_center([thigh, shank], axis=axis, initial=(core[0][0], core[1][0], core[2][0]))

    m = markers.get_rigid_body("ben:hip")
    x += [m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x]
    y += [m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y]
    z += [m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    m = markers.get_rigid_body("ben:RightThigh")
    x += [m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x]
    y += [m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y]
    z += [m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    m = markers.get_rigid_body("ben:RightShank")
    x += [m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x]
    y += [m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y]
    z += [m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    x_center = [ xc[0]]
    y_center =  [xc[1]]
    z_center = [xc[2]]

    ax.clear()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis([-500, 500, -750, 1500])
    ax.set_zlim3d(0, 1250)
    ax.scatter(x, y, z, c='r', marker='o')
    ax.scatter(x_center, y_center, z_center, c='g', marker='o')
    axis_x = [( axis[0] * 1000).item(0), (-axis[0] * 1000).item(0)]
    axis_y = [(axis[1] * 1000).item(0),  ( -axis[1] * 1000).item(0)]
    axis_z = [( axis[2] * 1000).item(0), ( -axis[2] * 1000).item(0)]
    ax.plot(axis_x, axis_y, axis_z, 'b')

ani = animation.FuncAnimation(fig, animate, interval=100)
# for i in xrange(1000):
#     animate(i)

plt.show()