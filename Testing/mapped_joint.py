from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Vicon import Markers
from Utilities import Mean_filter
import time
from Trajectories import center_of_rotation

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_autoscale_on(False)
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
angles = []
offset = 10

#centers, axis = center_of_rotation.leastsq_method(markers=markers)
#centers, axis, frame_index = center_of_rotation.sphere_method(markers=markers)
centers, axis = center_of_rotation.rotation_method(markers=markers)

x_center = []
y_center = []
z_center = []

for center in centers:
    x_center.append(center[0])
    y_center.append(center[1])
    z_center.append(center[2])

def animate(frame):
    count = frame
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
    #ax.scatter( [x_center[frame-offset]],[y_center[frame-offset]],[z_center[frame-offset]],  c='g', marker='o')
    ax.scatter([x_center[count]], [y_center[count]], [z_center[count ]], c='g', marker='o')
    # axis_x = [(core[0] - axis[0] * 1000).item(0), (core[0]).item(0), (core[0] + axis[0] * 1000).item(0)]
    # axis_y = [(core[1] - axis[1] * 1000).item(0), (core[1]).item(0), (core[1] + axis[1] * 1000).item(0)]
    # axis_z = [(core[2] - axis[2] * 1000).item(0), core[2].item(0), (core[2] + axis[2] * 1000).item(0)]
    #
    # ax.plot(axis_x, axis_y, axis_z, 'b')

ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()
np.savetxt("angles.csv", np.asarray(angles), delimiter=",")



