from Vicon import Vicon
from lib.Exoskeleton.Robot import core
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
from Vicon import Markers
from Utilities import Mean_filter
import time

frames = {}


frames["Root"] =    [Vicon.Point(0.0, 14.0, 0.0),
                     Vicon.Point(56.0, 0.0, 0.0),
                     Vicon.Point(14.0, 63.0, 0.0),
                     Vicon.Point(56.0, 63.0, 0.0)]

frames["R_Femur"] = [Vicon.Point(7.0, 0.0, 0.0),
                     Vicon.Point(56.0, 0.0, 0.0),
                     Vicon.Point(0.0, 70.0, 0.0),
                     Vicon.Point(42.0, 49.0, 0.0)]


frames["L_Femur"] = [Vicon.Point(0.0, 0.0, 0.0),
                     Vicon.Point(70.0, 0.0, 0.0),
                     Vicon.Point(0.0, 42.0, 0.0),
                     Vicon.Point(70.0, 56.0, 0.0)]


frames["R_Tibia"] = [Vicon.Point(0.0, 0.0, 0.0),
                     Vicon.Point(70.0, 0.0, 0.0),
                     Vicon.Point(0.0, 49.0, 0.0),
                     Vicon.Point(70.0, 63.0, 0.0)]


frames["L_Tibia"] = [Vicon.Point(0.0, 0.0, 0.0),
                     Vicon.Point(0.0, 63.0, 0.0),
                     Vicon.Point(70.0, 14.0, 0.0),
                     Vicon.Point(35.0, 49.0, 0.0)]

frames["L_Foot"] = [Vicon.Point(0.0, 0.0, 0.0),
                    Vicon.Point(70.0, 0.0, 0.0),
                    Vicon.Point(28.0, 70.0, 0.0),
                    Vicon.Point(70.0, 63.0, 0.0)]

frames["R_Foot"] = [Vicon.Point(0.0, 0.0, 0.0),
                    Vicon.Point(56.0, 0.0, 0.0),
                    Vicon.Point(14.0, 63.0, 0.0),
                    Vicon.Point(56.0, 63.0, 0.0)]



data = Vicon.Vicon("/home/nathanielgoldfarb/gait_analysis_toolkit/testing_data/advanced_lower_gait_cal_temp2.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_transform(frames)
joints = []
m = markers.get_rigid_body("R_Femur")

for frame in xrange(0, 3000):

    f = [m[0][frame], m[1][frame], m[2][frame], m[3][frame]]
    T, err = Markers.cloud_to_cloud(frames["R_Femur"], f)
    print frame, " ", err

global_r_hip, axis_r_hip, local_r_hip = markers.calc_joint_center("Root", "R_Femur", 300, 450)
r_hip = Markers.batch_transform_vector(markers.get_frame("Root"), local_r_hip)
joints.append(r_hip)

# global_l_hip, axis_l_hip, local_l_hip = markers.calc_joint_center("Root", "L_Femur", 1900, 2500)
# l_hip = Markers.batch_transform_vector(markers.get_frame("Root"), local_l_hip)
# joints.append(l_hip)

global_r_knee, axis_r_knee, local_r_knee = markers.calc_joint_center("Root", "R_Tibia", 1300, 1400)
r_knee = Markers.batch_transform_vector(markers.get_frame("Root"), local_r_knee)
joints.append(r_knee)

# global_l_knee, axis_l_knee, local_l_knee = markers.calc_joint_center("Root", "L_Tibia", 2800, 3000)
# l_knee = Markers.batch_transform_vector(markers.get_frame("Root"), local_l_knee)
# joints.append(l_knee)
# global_r_ankle, axis_r_ankle, local_r_ankle = markers.calc_joint_center("Root", "R_Foot", 1500, 1700)
# r_ankle = Markers.batch_transform_vector(markers.get_frame("R_Foot"), local_r_ankle)
# joints.append(r_ankle)
# global_l_ankle, axis_l_ankle, local_l_ankle = markers.calc_joint_center("Root", "L_Foot", 3100, 3300)
# l_ankle = Markers.batch_transform_vector(markers.get_frame("L_Foot"), local_l_ankle)
# joints.append(l_ankle)

markers.play(joints=joints)