import numpy as np
from lib.Exoskeleton.Robot import core
from Vicon import Vicon
from Vicon import Markers
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
fig = plt.figure()
ax = fig.gca(projection='3d')

data = Vicon.Vicon("/media/nathaniel/2855-CDF0/ben Cal 02.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
shank_markers = markers.get_rigid_body("ben:RightShank")[0:300]
thigh_markers = markers.get_rigid_body("ben:RightThigh")[0:300]
frame = 550

m1 = shank_markers[0][frame-1:frame+2]
m2 = shank_markers[1][frame-1:frame+2]
m3 = shank_markers[2][frame-1:frame+2]
m4 = shank_markers[3][frame-1:frame+2]
CoR = []
AoR = []
T_Th = markers.get_frame("ben:RightThigh")[0:300]
T_Sh = markers.get_frame("ben:RightShank")[0:300]
T_H = markers.get_frame("ben:hip")[0:300]
T_H_Sh = Markers.get_all_transformation_to_base(T_H, T_Sh)
T_H_Th = Markers.get_all_transformation_to_base(T_H, T_Th)
T_TH_SH = Markers.get_all_transformation_to_base(T_Th, T_Sh)
adjusted = Markers.transform_markers(np.linalg.inv(T_Th), shank_markers)

# for ii in xrange(2,1000):
#     current_points = []
#     for marker in shank_markers:
#         points = marker[ii:ii+2]
#         current_points.append(points)
#     center = Markers.calc_CoR(current_points)
#     axis = Markers.calc_AoR(current_points)
#     CoR.append(center)
#     AoR.append(axis)

markers = [m1, m2, m3, m4]
core = Markers.calc_CoR(markers)
axis = Markers.calc_AoR(markers)

x = []
y = []
z = []
i = 0
for marker in markers:

    for point in marker:
        x.append(point.x)
        y.append(point.y)
        z.append(point.z)
        ax.text(point.x,point.y,point.z, '%s' % (str(i)),
                size=20, zorder=1)

ax.scatter(x,y,z,'r*')

x = [ thigh_markers[0][frame].x, thigh_markers[1][frame].x, thigh_markers[2][frame].x, thigh_markers[3][frame].x ]
y = [ thigh_markers[0][frame].y, thigh_markers[1][frame].y, thigh_markers[2][frame].y, thigh_markers[3][frame].y ]
z = [ thigh_markers[0][frame].z, thigh_markers[1][frame].z, thigh_markers[2][frame].z, thigh_markers[3][frame].z]
ax.scatter(x,y,z,'r')
for i in xrange(4):
    ax.text(thigh_markers[i][frame].x, thigh_markers[i][frame].y, thigh_markers[i][frame].z,  '%s' % (str(i)), size=20, zorder=1)
axis_x = [(core[0]-axis[0]*500).item(0), (core[0]).item(0), (core[0]+axis[0]*500).item(0)]
axis_y = [(core[1]-axis[1]*500).item(0), (core[1]).item(0), (core[1]+axis[1]*500).item(0)]
axis_z = [(core[2]-axis[2]*500).item(0), core[2].item(0), (core[2]+axis[2]*500).item(0)]
ax.plot(axis_x,axis_y,axis_z,'r')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()