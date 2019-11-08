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
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()


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
    offset = 1
    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:]

    m1 = shank_markers[0][frame]
    m2 = shank_markers[1][frame]
    m3 = shank_markers[2][frame]
    m4 = shank_markers[3][frame]

    data = [m1, m2, m3, m4]
    T_Th = markers.get_frame("ben:RightThigh")[0:]
    T_Sh = markers.get_frame("ben:RightShank")[0:]

    T_TH_SH_1 = np.dot(np.linalg.pinv(T_Th[frame]), T_Sh[frame])  # Markers.get_all_transformation_to_base(T_Th, T_Sh)[frame]
    T_TH_SH_2 = np.dot(np.linalg.pinv(T_Th[frame + offset]), T_Sh[frame + offset])

    R1 = T_TH_SH_1[:3, :3]
    R2 = T_TH_SH_2[:3, :3]
    R1_2 = np.dot(np.transpose(R2), R1)

    axis, theta = Markers.R_to_axis_angle(R2)
    axis[0], axis[2] = axis[2], axis[0]

    rp_1 = Markers.calc_mass_vect(
        [shank_markers[0][frame], shank_markers[1][frame], shank_markers[2][frame], shank_markers[3][frame]])
    rp_2 = Markers.calc_mass_vect(
        [shank_markers[0][frame + offset], shank_markers[1][frame + offset], shank_markers[2][frame + offset],
         shank_markers[3][frame + offset]])

    rd_1 = Markers.calc_mass_vect(
        [thigh_markers[0][frame], thigh_markers[1][frame], thigh_markers[2][frame], thigh_markers[3][frame]])
    rd_2 = Markers.calc_mass_vect(
        [thigh_markers[0][frame + offset], thigh_markers[1][frame + offset], thigh_markers[2][frame + offset],
         thigh_markers[3][frame + offset]])

    rdp1 = np.dot(T_Sh[frame][:3, :3], rd_1 - rp_1)
    rdp2 = np.dot(T_Sh[frame + offset][:3, :3], rd_2 - rp_2)

    P = np.eye(3) - R1_2
    Q = rdp2 - np.dot(R1_2, rdp1)

    rc = np.dot(np.linalg.pinv(P), Q)
    Rc = rp_1 + np.dot(np.transpose(T_Sh[frame][:3, :3]), rc)

    thigh = Markers.calc_mass_vect([thigh_markers[0][frame],
                                    thigh_markers[1][frame],
                                    thigh_markers[2][frame],
                                    thigh_markers[3][frame]])

    shank = Markers.calc_mass_vect([shank_markers[0][frame],
                                    shank_markers[1][frame],
                                    shank_markers[2][frame],
                                    shank_markers[3][frame]])

    sol = Markers.minimize_center([thigh, shank], axis=axis, initial=(Rc[0], Rc[1], Rc[2]))
    Rc = sol.x

    axis_x = [(Rc[0] - axis[0] * 1000).item(0), (Rc[0]).item(0), (Rc[0] + axis[0] * 1000).item(0)]
    axis_y = [(Rc[1] - axis[1] * 1000).item(0), (Rc[1]).item(0), (Rc[1] + axis[1] * 1000).item(0)]
    axis_z = [(Rc[2] - axis[2] * 1000).item(0), Rc[2].item(0), (Rc[2] + axis[2] * 1000).item(0)]

    ax.plot(axis_x, axis_y, axis_z, 'b')
    ax.scatter(Rc[0], Rc[1], Rc[2], 'b')


ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()