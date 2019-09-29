
from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_autoscale_on(False)
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ridgid_markers.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
shank_markers = markers.get_rigid_body("LowerLegs_Healthy:RightShank")[0:300]
T_Th = markers.get_frame("LowerLegs_Healthy:RightThigh")[0:300]
T_Sh = markers.get_frame("LowerLegs_Healthy:RightShank")[0:300]
T_H = markers.get_frame("LowerLegs_Healthy:Hip")[0:300]
T_H_Sh = rigid_marker.get_all_transformation_to_base(T_H, T_Sh)
T_H_Th = rigid_marker.get_all_transformation_to_base(T_H, T_Th)
T_TH_SH = rigid_marker.get_all_transformation_to_base(T_Th, T_Sh)

adjusted = rigid_marker.transform_markers( np.linalg.inv(T_Th), shank_markers)
CoR = []
for ii in xrange(2,275):
    current_points = []
    for marker in adjusted:
        points = marker[ii-1:ii+2]
        current_points.append(points)
    T = T_H[ii]
    center = rigid_marker.find_CoR(current_points)
    CoR.append(center)

adjusted = rigid_marker.transform_markers( np.linalg.inv(T_Th), shank_markers)

def animate(frame):

    x = []
    y = []
    z = []
    m = markers.get_rigid_body("LowerLegs_Healthy:Hip")[0:300]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')

    m = markers.get_rigid_body("LowerLegs_Healthy:RightThigh")[0:300]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')

    m = markers.get_rigid_body("LowerLegs_Healthy:RightShank")[0:300]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')
    # #
    m = markers.get_rigid_body("LowerLegs_Healthy:RightFoot")[0:300]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')

    m = markers.get_rigid_body("LowerLegs_Healthy:LeftThigh")[0:300]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')

    m = markers.get_rigid_body("LowerLegs_Healthy:LeftShank")[0:300]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]
    #ax.scatter(x, y, z, c='r', marker='o')
    # #
    m = markers.get_rigid_body("LowerLegs_Healthy:LeftFoot")[0:300]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    ax.clear()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis([-500, 500, -2000, 3000])
    ax.scatter(x, y, z, c='r', marker='o')
    cor = CoR[frame]
    #ax.scatter(cor[0], cor[1], cor[2], c='b', marker='o')


ani = animation.FuncAnimation(fig, animate, interval=10)
plt.show()