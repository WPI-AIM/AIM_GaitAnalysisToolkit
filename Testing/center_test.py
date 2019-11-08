from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Vicon import Markers
from Utilities import Mean_filter
import time
from scipy import signal
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n



data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
shank_markers = markers.get_rigid_body("ben:RightShank")[0:1000]
offset = 5
T_Th = markers.get_frame("ben:RightThigh")
T_Sh = markers.get_frame("ben:RightShank")
adjusted = rigid_marker.transform_markers(np.linalg.inv(T_Th), shank_markers)

radii = []
x = []
y = []
z = []

for frame in xrange(offset, len(adjusted[0])-offset):

    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:]

    T_Th = markers.get_frame("ben:RightThigh")[0:]
    T_Sh = markers.get_frame("ben:RightShank")[0:]

    T_TH_SH_1 = np.dot(np.linalg.pinv(T_Th[frame]), T_Sh[frame])  # Markers.get_all_transformation_to_base(T_Th, T_Sh)[frame]
    T_TH_SH_2 = np.dot(np.linalg.pinv(T_Th[frame + offset]), T_Sh[frame + offset])
    R1 = T_TH_SH_1[:3, :3]
    R2 = T_TH_SH_2[:3, :3]
    R1_2 = np.dot(np.transpose(R2), R1)

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

    axis, angle = Markers.R_to_axis_angle(T_TH_SH_1[0:3, 0:3])

    sol = Markers.minimize_center([thigh, shank], axis=axis, initial=(rc[0], rc[1], rc[2]))
    Rc = sol.x
    x.append(Rc[0])
    y.append(Rc[1])
    z.append(Rc[2])

fs = 100.0
fc = 10.0  # Cut-off frequency of the filter
w = fc / (fs / 2.) # Normalize the frequency
print
b, a = signal.butter(30, w, 'low')
output = signal.filtfilt(b, a, z,axis=0)
N = 10
output = moving_average(output, 10)
#plt.plot(radii)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z, c='r', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
plt.show()
