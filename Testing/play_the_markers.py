from Vicon import Vicon
import numpy as np
import matplotlib.pyplot as plt
from Trajectories import center_of_rotation
from lib.Exoskeleton.Robot import core
from scipy import signal
from Vicon import Markers

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
global_joint, axis, local_joint = markers.calc_joint_center("RightThigh", "RightShank", 200, 325)
Tp = markers.get_frame("RightThigh")
# all_global = []
#
# for T in Tp:
#     p = np.dot(T, local_joint )
#     all_global.append(p)

all_global = Markers.batch_transform_vector(Tp, local_joint)

markers.play([all_global], save=True)
# # TODO need to get all the joint centers now using the new rebase



# x = []
# y = []
# z = []
#
# for c in core:
#     x.append(c[0].item(0))
#     y.append(c[1].item(0))
#     z.append(c[2].item(0))
#
# # fs = 100.0
# # fc = 10.0  # Cut-off frequency of the filter
# # w = fc / (fs / 2.) # Normalize the frequency
# # print
# # b, a = signal.butter(30, w, 'low')
# # x = signal.filtfilt(b, a, x, axis=0)
# # y = signal.filtfilt(b, a, y, axis=0)
# # z = signal.filtfilt(b, a, z, axis=0)
# # N = 10
#
#
# def smooth(y, box_pts):
#     box = np.ones(box_pts)/box_pts
#     y_smooth = np.convolve(y, box, mode='same')
#     return y_smooth
#
# fig, ax = plt.subplots()
#
# ax.plot(x, 'r-')
# ax.plot(y, 'r-')
# ax.plot(z, 'r-')
#
# plt.plot(smooth(x, 19), 'g-')
# plt.plot(smooth(y, 19), 'g-')
# plt.plot(smooth(z, 19), 'g-')
#
#
# plt.show()
#centers = [core]
#markers.play([center])

