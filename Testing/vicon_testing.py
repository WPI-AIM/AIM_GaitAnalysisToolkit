from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ridgid_markers.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
thigh_markers = markers.get_rigid_body("LowerLegs_Healthy:RightThigh")[0:300]
shank_markers = markers.get_rigid_body("LowerLegs_Healthy:RightShank")[0:300]
print shank_markers

T_Th =  markers.get_frame("LowerLegs_Healthy:RightThigh")[0:300]
T_Sh = markers.get_frame("LowerLegs_Healthy:RightShank")[0:300]
T_H = markers.get_frame("LowerLegs_Healthy:Hip")[0:300]
T_H_Sh = rigid_marker.get_all_transformation_to_base(T_H, T_Sh)
T_H_Th = rigid_marker.get_all_transformation_to_base(T_H, T_Th)
T_TH_SH = rigid_marker.get_all_transformation_to_base(T_Th, T_Sh)

adjusted = rigid_marker.transform_markers( np.linalg.inv(T_Th), shank_markers)
CoR = []


for ii in xrange(2, 250):
    current_points = []
    for marker in adjusted:
        points = marker[ii:ii+2]
        current_points.append(points)
    center = rigid_marker.find_CoR(current_points)

    CoR.append( np.sqrt(np.sum(center**2)))
#
# shank = []
# for vect in CoR:
#     v = np.vstack([vect, 1])
#     v_prime = np.dot(T_TH_SH,v)
#     shank.append(v_prime[0,0:3])
#
# angles = []
# for vect1, vect2 in zip(CoR, shank):
#     print "vect1 ", vect1.T[0]
#     print "vect2 ",  vect2.T[0]
#     angle = rigid_marker.get_angle_between_vects(vect1.T[0],vect2.T[0])
#     angles.append(angle)

plt.plot(CoR)
plt.show()







