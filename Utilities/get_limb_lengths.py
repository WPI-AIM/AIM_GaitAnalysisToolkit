from Vicon import Vicon
import numpy as np
import matplotlib.pyplot as plt
from Trajectories import center_of_rotation
from lib.Exoskeleton.Robot import core
from scipy import signal

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


frames = {}
frames["hip"] = [core.Point(0.0, 0.0, 0.0),
                 core.Point(70.0, 0, 0.0),
                 core.Point(0, 42.0, 0),
                 core.Point(35.0, 70.0, 0.0)]

frames["RightThigh"] = [core.Point(0.0, 0.0, 0.0),
                        core.Point(56.0, 0, 0.0),
                        core.Point(0, 49.0, 0),
                        core.Point(56.0, 63.0, 0.0)]

frames["RightShank"] = [core.Point(0.0, 0.0, 0.0),
                        core.Point(56.0, 0, 0.0),
                        core.Point(0, 42.0, 0),
                        core.Point(56.0, 70.0, 0.0)]

data = Vicon.Vicon("/home/nathanielgoldfarb/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_transform(frames)
core, axis = markers.calc_joint_center("RightThigh", "RightShank", 200, 325)
x = []
y = []
z = []

for c in core:
    x.append(c[0].item(0))
    y.append(c[1].item(0))
    z.append(c[2].item(0))

# fs = 100.0
# fc = 10.0  # Cut-off frequency of the filter
# w = fc / (fs / 2.) # Normalize the frequency
# print
# b, a = signal.butter(30, w, 'low')
# x = signal.filtfilt(b, a, x, axis=0)
# y = signal.filtfilt(b, a, y, axis=0)
# z = signal.filtfilt(b, a, z, axis=0)
# N = 10


def smooth(y, box_pts):
    box = np.ones(box_pts)/box_pts
    y_smooth = np.convolve(y, box, mode='same')
    return y_smooth

fig, ax = plt.subplots()

ax.plot(x, 'r-')
ax.plot(y, 'r-')
ax.plot(z, 'r-')

plt.plot(smooth(x, 19), 'g-')
plt.plot(smooth(y, 19), 'g-')
plt.plot(smooth(z, 19), 'g-')


plt.show()
#centers = [core]
#markers.play([center])

