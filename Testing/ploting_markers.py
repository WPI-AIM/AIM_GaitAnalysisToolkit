
from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Vicon import Markers

import time

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_autoscale_on(False)
data = Vicon.Vicon("/media/nathaniel/2855-CDF0/ben Cal 02.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
shank_markers = markers.get_rigid_body("ben:RightShank")[0:1000]
thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:1000]
T_Th = markers.get_frame("ben:RightThigh")[0:1000]
T_Sh = markers.get_frame("ben:RightShank")[0:1000]
T_H = markers.get_frame("ben:hip")[0:1000]
T_H_Sh = rigid_marker.get_all_transformation_to_base(T_H, T_Sh)
T_H_Th = rigid_marker.get_all_transformation_to_base(T_H, T_Th)
T_TH_SH = rigid_marker.get_all_transformation_to_base(T_Th, T_Sh)

adjusted = rigid_marker.transform_markers( np.linalg.inv(T_H), thigh_markers)
CoR = []
AoR = []
for ii in xrange(2,800):
    current_points = []
    for marker in shank_markers:
        points = marker[ii:ii+2]
        current_points.append(points)
    T = T_H[ii]
    center = np.append(Markers.calc_CoR(current_points), 1)
    axis = np.append(Markers.calc_AoR(current_points), 1)
    # center = np.dot(T, np.array(center).reshape((-1, 1)))
    # axis = np.dot(T, np.array(axis).reshape((-1, 1)))
    CoR.append(center[0:3])
    AoR.append(axis[0:3])


def animate(frame):
    frame = frame +1
    x = []
    y = []
    z = []

    m = markers.get_rigid_body("ben:hip")[0:1000]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')

    m = markers.get_rigid_body("ben:RightThigh")[0:1000]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')

    m = markers.get_rigid_body("ben:RightShank")[0:1000]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')
    # # #
    # m = markers.get_rigid_body("ben:RightFoot")[0:1000]
    # x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    # y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    # z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    # #ax.scatter(x, y, z, c='r', marker='o')
    #
    # m = markers.get_rigid_body("ben:LeftThigh")[0:1000]
    # x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    # y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    # z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    # #ax.scatter(x, y, z, c='r', marker='o')
    #
    # m = markers.get_rigid_body("ben:LeftShank")[0:1000]
    # x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    # y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    # z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    # #ax.scatter(x, y, z, c='r', marker='o')
    # # #
    # m = markers.get_rigid_body("ben:LeftFoot")[0:1000]
    # x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    # y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    # z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    ax.clear()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis([-400, 400, -2200, 3200])
    ax.set_zlim3d(0,1250)
    ax.scatter(x, y, z, c='r', marker='o')

    shank_markers = markers.get_rigid_body("ben:RightShank")[0:1000]

    m1 = shank_markers[0][frame-1:frame + 3]
    m2 = shank_markers[1][frame-1:frame + 3]
    m3 = shank_markers[2][frame-1:frame + 3]
    m4 = shank_markers[3][frame-1:frame + 3]
    data = [m1, m2, m3, m4]
    loc = m1[0]
    core =  Markers.calc_CoR(data)
    axis =  Markers.calc_AoR(data)
    axis_x = [(core[0] - axis[0] * 1000).item(0), (core[0]).item(0), (core[0] + axis[0] * 1000).item(0)]
    axis_y = [(core[1] - axis[1] * 1000).item(0), (core[1]).item(0), (core[1] + axis[1] * 1000).item(0)]
    axis_z = [(core[2] - axis[2] * 1000).item(0), core[2].item(0), (core[2] + axis[2] * 1000).item(0)]
    # axis_x = [core[0]]
    # axis_y = [core[1]]
    # axis_z = [core[2]]
    ax.plot(axis_x, axis_y, axis_z, 'b')


ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()