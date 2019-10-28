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

#centers, axes = center_of_rotation.leastsq_method(markers=markers)
centers, axes = center_of_rotation.sphere_method(markers=markers)
center_of_rotation.projection(markers)
#centers, axes = center_of_rotation.sphere_method2(markers=markers)

#centers, axes = center_of_rotation.rotation_method2(markers=markers)
#centers, axes = center_of_rotation.rotation_method2(markers=markers)
#centers, axes = center_of_rotation.leastsq_method2(markers=markers)

x = []
y = []
z = []

for center in centers:
    print center
    x.append(center[0])
    y.append(center[1])
    z.append(center[2])


# fs = 100.0
# fc = 30.0  # Cut-off frequency of the filter
# w = fc / (fs / 2.) # Normalize the frequency
# b, a = signal.butter(30, w, 'low')
#
# x = signal.filtfilt(b, a, x)
# x = moving_average(x, 20)
#
# y = signal.filtfilt(b, a, y)
# y = moving_average(y, 20)
#
# z = signal.filtfilt(b, a, z)
# z = moving_average(z, 20)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.axis([-500, 500, -750, 1500])
ax.set_zlim3d(0, 1250)

ax.scatter(x, y, z, c='r', marker='o')
# print x

for ii, axis in enumerate(axes):
    axis_x = [(x[ii]+ axis[0] * 1000).item(0), x[ii].item(0), (x[ii]-axis[0] * 1000).item(0)]
    axis_y = [(y[ii] +axis[1] * 1000).item(0), y[ii].item(0), (y[ii]-axis[1] * 1000).item(0)]
    axis_z = [(z[ii]+axis[2] * 1000).item(0), z[ii].item(0) , (z[ii]-axis[2] * 1000).item(0)]

    ax.plot(axis_x, axis_y, axis_z, 'b')


x = []
y = []
z = []
frame = 10
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
ax.scatter(x, y, z, c='g', marker='o')

ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')

plt.show()
