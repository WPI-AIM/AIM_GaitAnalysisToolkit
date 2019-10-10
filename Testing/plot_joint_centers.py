from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Vicon import Markers
from Trajectories import center_of_rotation
from scipy import signal

def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()

#centers, axis = center_of_rotation.leastsq_method(markers=markers)
#centers, axis = center_of_rotation.sphere_method(markers=markers)
centers, axis = center_of_rotation.rotation_method(markers=markers)

x = []
y = []
z = []

for center in centers:
    print center
    x.append(center[0])
    y.append(center[1])
    z.append(center[2])


fs = 100.0
fc = 30.0  # Cut-off frequency of the filter
w = fc / (fs / 2.) # Normalize the frequency
b, a = signal.butter(30, w, 'low')

x = signal.filtfilt(b, a, x)
x = moving_average(x, 20)

y = signal.filtfilt(b, a, y)
y = moving_average(y, 20)

z = signal.filtfilt(b, a, z)
z = moving_average(z, 20)

plt.plot(x)
plt.plot(y)
plt.plot(z)
plt.legend(["x", "y", "z"])
plt.show()
