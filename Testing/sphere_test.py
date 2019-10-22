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
offset = 10
T_Th = markers.get_frame("ben:RightThigh")
adjusted = rigid_marker.transform_markers(np.linalg.inv(T_Th), shank_markers)
radii = []
x = []
y = []
z = []

for frame_number in xrange(offset, len(adjusted[0])-offset):

    m1 = adjusted[0][frame_number-offset:frame_number+offset+1]
    m2 = adjusted[1][frame_number-offset:frame_number+offset+1]
    m3 = adjusted[2][frame_number-offset:frame_number+offset+1]
    m4 = adjusted[3][frame_number-offset:frame_number+offset+1]

    data = [m1, m2, m3, m4]
    raduis, C = Markers.sphereFit(data)
    T_Th = markers.get_frame("ben:RightThigh")[frame_number]
    T_Sh = markers.get_frame("ben:RightShank")[frame_number]
    T = np.dot(np.linalg.pinv(T_Th), T_Sh)


    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]
    thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:]

    thigh = Markers.calc_mass_vect([thigh_markers[0][frame_number],
                                    thigh_markers[1][frame_number],
                                    thigh_markers[2][frame_number],
                                    thigh_markers[3][frame_number]])

    shank = Markers.calc_mass_vect([shank_markers[0][frame_number],
                                    shank_markers[1][frame_number],
                                    shank_markers[2][frame_number],
                                    shank_markers[3][frame_number]])

    axis, angle = Markers.R_to_axis_angle(T[0:3, 0:3])
    axis[0], axis[2] = axis[2], axis[0]
    sol = Markers.minimize_center([thigh, shank], axis=axis, initial=(C[0][0], C[1][0], C[2][0]))
    x.append(C[0])
    y.append(C[1])
    z.append(C[2])
    radii.append(raduis)

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
#plt.plot(output)
plt.plot(z)
plt.show()
