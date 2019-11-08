from Vicon import Vicon
import numpy as np
import math
import matplotlib.pyplot as plt

def transform_btw_two_frames(from_frame, to_frame):
    return np.matmul(np.linalg.inv(from_frame), to_frame)

def rotation_angle_from_transformation_mat(trans):
    return 0


# TODO
def rotation_axis_from_transformation_mat(trans):
    return [1, 0, 0]


def distance_from_transformation_mat(trans):
    return math.sqrt(pow(trans[0][3], 2) + pow(trans[1][3], 2) + pow(trans[2][3], 2))



data = Vicon.Vicon("/home/benjamin/PycharmProjects/Exo/gait_analysis_toolkit/testing_data/ridgid_markers.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
thigh = markers.get_rigid_body("LowerLegs_Healthy:RightThigh")
shank = markers.get_rigid_body("LowerLegs_Healthy:RightShank")

print markers.get_frame("LowerLegs_Healthy:RightThigh")[0]
print markers.get_frame("LowerLegs_Healthy:RightShank")[0]

print len(markers.get_frame("LowerLegs_Healthy:RightThigh"))

Tth_sh0 = transform_btw_two_frames(markers.get_frame("LowerLegs_Healthy:RightThigh")[0], markers.get_frame("LowerLegs_Healthy:RightShank")[0])
print Tth_sh0
ang_th_sh0 = rotation_angle_from_transformation_mat(Tth_sh0)

radius = np.array([])
axis_thigh_shank = np.array([])
angle = np.array([])
dists = []
for idx in range(len(markers.get_frame("LowerLegs_Healthy:RightThigh")) - 1):
    try: # if the transformation matrices are correct, this try will work
        trans_th1_sh1 = transform_btw_two_frames(markers.get_frame("LowerLegs_Healthy:RightThigh")[idx], markers.get_frame("LowerLegs_Healthy:RightShank")[idx])
        trans_th2_sh2 = transform_btw_two_frames(markers.get_frame("LowerLegs_Healthy:RightThigh")[idx + 1], markers.get_frame("LowerLegs_Healthy:RightShank")[idx + 1])
        trans_sh1_sh2 = transform_btw_two_frames(trans_th1_sh1, trans_th2_sh2)
        dist = distance_from_transformation_mat(trans_sh1_sh2)
        axis = rotation_axis_from_transformation_mat(trans_sh1_sh2)
        ang = rotation_angle_from_transformation_mat(trans_sh1_sh2)
        radius = np.append(radius, (dist/2.0) * math.sin(ang/2.0))
        axis_thigh_shank = np.append(axis_thigh_shank, axis)
        angle = np.append(angle, rotation_angle_from_transformation_mat(transform_btw_two_frames(markers.get_frame("LowerLegs_Healthy:RightThigh")[idx], markers.get_frame("LowerLegs_Healthy:RightShank")[idx])) - ang_th_sh0)
        dists.append(dist)
    except: # if the transformation matrices are incomplete, skip them
        pass
print dists
plt.plot(dists)
plt.show()


# create equation for radius(angle)
# plot


# Get frames for thigh and shank
# Define thigh frame as base frame
# Find transformation from thigh to shank, which will include original rotation
# For consecutive frames:
#   get transformation between shank frame i and shank frame i+1
#       get distance between shank frame i and shank frame i+1
#       get axis of rotation between shank frame i and shank frame i+1
#       get angle of rotation between shank frame i and shank frame i+1
#       assume radius for shank frame i and shank frame i+1 to the axis of rotation is the same
#           (for consecutive frames the difference will be small)
#       radius = (distance/2)*sin(theta/2)
# get transformation from thigh to shank
#   get rotation angle, minus original rotation
# Create angle-radius pairs
# Find coefficients of equation at^4 + bt^3 + ct^2 + dt + e = r(t) that best match data
# record and display axis of rotation vs angle
