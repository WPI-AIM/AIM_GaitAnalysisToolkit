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
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
cor_filter = Mean_filter.Mean_Filter(20)
cor_filter2 = Mean_filter.Mean_Filter(20)
all_core = []

angles = []

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
    ax.axis([-200, 200, 0, 1000])
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

    T_Th = markers.get_frame("ben:RightThigh")[frame]
    T_Sh = markers.get_frame("ben:RightShank")[frame]
    T = np.dot(np.linalg.pinv(T_Th), T_Sh)
    axis, angle = Markers.R_to_axis_angle(T[0:3, 0:3])

    thigh = Markers.calc_mass_vect([thigh_markers[0][frame],
                                    thigh_markers[1][frame],
                                    thigh_markers[2][frame],
                                    thigh_markers[3][frame]])

    shank = Markers.calc_mass_vect([shank_markers[0][frame],
                                    shank_markers[1][frame],
                                    shank_markers[2][frame],
                                    shank_markers[3][frame]])
    axis[0], axis[2] = axis[2], axis[0]
    sol = Markers.minimize_center([thigh, shank], axis=axis, initial=(core[0][0], core[1][0], core[2][0]))
    temp_center = sol.x
    temp_center = cor_filter2.update(temp_center)

    axis_x = [(temp_center[0] - axis[0] * 1000).item(0), temp_center[0].item(0), (temp_center[0] + axis[0] * 1000).item(0)]
    axis_y = [(temp_center[1] - axis[1] * 1000).item(0), temp_center[1].item(0), (temp_center[1] + axis[1] * 1000).item(0)]
    axis_z = [(temp_center[2] - axis[2] * 1000).item(0), temp_center[2].item(0), (temp_center[2] + axis[2] * 1000).item(0)]
    th = thigh - temp_center[:3]
    sh = shank - temp_center[:3]

    dist_shank = np.sqrt(np.sum(np.power(sh, 2)))
    dist_thigh = np.sqrt(np.sum(np.power(th, 2)))

    angle = Markers.get_angle_between_vects(thigh - temp_center[:3], shank - temp_center[:3])
    angles.append([angle, dist_shank])

    ax.scatter( [temp_center[0]], [temp_center[1]], [temp_center[2]], 'go')
    ax.plot(axis_x, axis_y, axis_z, 'b')

    #ax.scatter(Rc[0], Rc[1], Rc[2], 'b')


ani = animation.FuncAnimation(fig, animate, interval=10)

plt.show()
np.savetxt("angles.csv", np.asarray(angles), delimiter=",")