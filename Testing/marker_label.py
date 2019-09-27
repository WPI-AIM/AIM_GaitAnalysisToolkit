from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ridgid_markers.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
thigh_markers = markers.get_rigid_body("LowerLegs_Healthy:RightThigh")[0:300]
shank_markers = markers.get_rigid_body("LowerLegs_Healthy:RightShank")[0:300]
print shank_markers
x = []
y = []
z = []
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
m = markers.get_rigid_body("LowerLegs_Healthy:RightShank")[0:300]
frame = 30

x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
ax.scatter(x, y, z, c='r', marker='o')
for i in xrange(4):
    ax.text(m[i][frame].x, m[i][frame].y, m[i][frame].z,  '%s' % (str(i)), size=20, zorder=1, color='k')
ax.set_xlabel('X Label')
ax.set_ylabel('Y Label')
ax.set_zlabel('Z Label')
plt.show()
