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
shank_markers = markers.get_rigid_body("ben:RightShank")[0:1000]
offset = 5
T_Th = markers.get_frame("ben:RightThigh")
adjusted = rigid_marker.transform_markers(np.linalg.inv(T_Th), shank_markers)

for frame in xrange(offset, len(adjusted[0])-offset):
    m1 = adjusted[0][frame - offset:frame + offset + 1]
    m2 = adjusted[1][frame - offset:frame + offset + 1]
    m3 = adjusted[2][frame - offset:frame + offset + 1]
    m4 = adjusted[3][frame - offset:frame + offset + 1]
    data = [m1, m2, m3, m4]
    core = Markers.calc_CoR(data)
    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:]
    T_Th = markers.get_frame("ben:RightThigh")[frame]
    T_Sh = markers.get_frame("ben:RightShank")[frame]
    T = np.dot(np.linalg.pinv(T_Th), T_Sh)
    thigh = Markers.calc_mass_vect([thigh_markers[0][frame],
                                    thigh_markers[1][frame],
                                    thigh_markers[2][frame],
                                    thigh_markers[3][frame]])

    shank = Markers.calc_mass_vect([shank_markers[0][frame],
                                    shank_markers[1][frame],
                                    shank_markers[2][frame],
                                    shank_markers[3][frame]])

    axis, angle = Markers.R_to_axis_angle(T[0:3, 0:3])
    axis[0], axis[2] = axis[2], axis[0]
    sol = Markers.minimize_center([thigh, shank], axis=axis, initial=(core[0][0], core[1][0], core[2][0]))
    x.append(core[0])
    y.append(core[1])
    z.append(core[2])

fs = 100.0
fc = 10.0  # Cut-off frequency of the filter
w = fc / (fs / 2.) # Normalize the frequency
print
b, a = signal.butter(30, w, 'low')
output = signal.filtfilt(b, a, z,axis=0)
N = 10
output = moving_average(output, 10)
#plt.plot(radii)
plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.show()
