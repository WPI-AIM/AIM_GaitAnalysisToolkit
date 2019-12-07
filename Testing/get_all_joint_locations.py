from Vicon import Vicon
import numpy as np
import matplotlib.pyplot as plt
from Trajectories import center_of_rotation
from lib.Exoskeleton.Robot import core
from Vicon import Vicon
from Vicon import Markers

from scipy import signal


def get_right_knee(file, start, end):
    vicon = Vicon.Vicon(file)
    markers = vicon.get_markers()
    frames = {}
    adjusted = Markers.transform_markers(np.linalg.inv(T_Th), shank_markers)

    markers.smart_sort()
    shank_markers = markers.get_rigid_body(joint)

    m1 = shank_markers[0][start:end]
    m2 = shank_markers[1][start:end]
    m3 = shank_markers[2][start:end]
    m4 = shank_markers[3][start:end]
    data = [m1, m2, m3, m4]

    core = Markers.calc_CoR(data)
    axis = Markers.calc_AoR(data)

    centers = []
    for i in xrange(len(shank_markers[0])):
        centers.append(core)

    return centers


def get_joint_location(file, joint, start, end):

    vicon = Vicon.Vicon(file)
    markers = vicon.get_markers()
    frames = {}

    markers.smart_sort()
    shank_markers = markers.get_rigid_body(joint)

    m1 = shank_markers[0][start:end]
    m2 = shank_markers[1][start:end]
    m3 = shank_markers[2][start:end]
    m4 = shank_markers[3][start:end]
    data = [m1, m2, m3, m4]

    core = Markers.calc_CoR(data)
    axis = Markers.calc_AoR(data)

    centers = []
    for i in xrange(len(shank_markers[0])):
       centers.append(core)


    return centers


def play(file):


    data = Vicon.Vicon(file)
    markers = data.get_markers()
    markers.smart_sort()
    markers.play()


if __name__ == "__main__":
    file = "/home/nathanielgoldfarb/Documents/Mocap_Participant/MoCap_Participants/subject_03/subject_03 Cal 03.csv"
    #play(file)

    right_hip = get_joint_location(file, "R_Femur", 260, 750) # 750
    right_knee = get_joint_location(file, "R_Tibia", 900, 950) # 950
    right_ankle = get_joint_location(file, "R_Foot", 1100, 1400)

    # left_hip = get_joint_location(file, "L_Femur", 1600, 2000)
    # left_knee = get_joint_location(file, "L_Tibia", 2200, 2300)
    # left_ankle = get_joint_location(file, "L_Foot", 2500, 2600)

    joints = [ right_hip, right_knee, right_ankle]

    data = Vicon.Vicon(file)
    markers = data.get_markers()
    markers.smart_sort()
    markers.play(joints)
