
from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Vicon import Markers

import time

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_autoscale_on(False)
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_walking01.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()

def animate(frame):
    frame = frame +1
    x = []
    y = []
    z = []

    m = markers.get_rigid_body("ben:hip")[0:]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    m = markers.get_rigid_body("ben:RightThigh")[0:]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]

    m = markers.get_rigid_body("ben:RightShank")[0:]
    x += [ m[0][frame].x, m[1][frame].x, m[2][frame].x, m[3][frame].x ]
    y += [ m[0][frame].y, m[1][frame].y, m[2][frame].y, m[3][frame].y ]
    z += [ m[0][frame].z, m[1][frame].z, m[2][frame].z, m[3][frame].z]



    ax.clear()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis([-400, 400, -2200, 3200])
    ax.set_zlim3d(0,1250)
    ax.scatter(x, y, z, c='r', marker='o')

    shank_markers = markers.get_rigid_body("ben:RightShank")[0:]

    m1 = shank_markers[0][frame:frame + 2]
    m2 = shank_markers[1][frame:frame + 2]
    m3 = shank_markers[2][frame:frame + 2]
    m4 = shank_markers[3][frame:frame + 2]
    data = [m1, m2, m3, m4]
    T_Th = markers.get_frame("ben:RightThigh")[0:]
    T_Sh = markers.get_frame("ben:RightShank")[0:]
    T_H = markers.get_frame("ben:hip")[0:]
    T_TH_SH =  np.dot(np.linalg.pinv(T_Th[frame]), T_Sh[frame])  #Markers.get_all_transformation_to_base(T_Th, T_Sh)[frame]
    print 'T_Th', T_Th[frame]
    print "---------------------"
    print 'T_Sh', T_Sh[frame]
    print "---------------------"
    # print "T_Th_SH",   T_TH_SH
    # print "---------------------"
    R, t = Markers.get_transformation(data)
    # print T_TH_SH[frame]
    center = Markers.get_center(data, R) - t
    vect  = np.array( (center[0].item(0), center[1].item(0), center[2].item(0),1  )).reshape((-1,1))
    center = np.dot( T_Sh[frame] , vect)

    # center = np.dot(T_TH_SH,center)
    # axis_x = [(core[0] - axis[0] * 1000).item(0), (core[0]).item(0), (core[0] + axis[0] * 1000).item(0)]
    # axis_y = [(core[1] - axis[1] * 1000).item(0), (core[1]).item(0), (core[1] + axis[1] * 1000).item(0)]
    # axis_z = [(core[2] - axis[2] * 1000).item(0), core[2].item(0), (core[2] + axis[2] * 1000).item(0)]

    axis_x = [center[0]]
    axis_y = [center[1]]
    axis_z = [center[2]]
    ax.scatter(axis_x, axis_y, axis_z, 'b')


ani = animation.FuncAnimation(fig, animate, interval=100)
plt.show()