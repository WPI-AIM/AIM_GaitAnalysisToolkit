from Vicon import Vicon
import numpy as np
import matplotlib.pyplot as plt
from Trajectories import center_of_rotation
from lib.Exoskeleton.Robot import core
from Vicon import Vicon
from Vicon import Markers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation
from scipy import signal

cloud = [core.Point(0.0, 0.0, 0.0),
         core.Point(70.0, 0.0, 0.0),
         core.Point(0.0, 49.0, 0.0),
         core.Point(70.0, 63.0, 0.0)]


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

ax.set_autoscale_on(False)

def animate(frame, x, y, z, centers=None, axis =None):
    """

    :param frame:
    :param x:
    :param y:
    :param z:
    :param centers:
    :return:
    """
    print frame
    ax.clear()
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis([-500, 500, -200, 3000])
    ax.set_zlim3d(0, 1250)
    ax.scatter(x[frame], y[frame], z[frame], c='r', marker='o')
    ax.scatter(centers[frame][0], centers[frame][1], centers[frame][2], c='g', marker='o')
    axis_x = [(centers[frame][0] - axis[0] * 1000).item(0), (centers[frame][0]).item(0), (centers[frame][0] + axis[0] * 1000).item(0)]
    axis_y = [(centers[frame][1] - axis[1] * 1000).item(0), (centers[frame][1]).item(0), (centers[frame][1] + axis[1] * 1000).item(0)]
    axis_z = [(centers[frame][2] - axis[2] * 1000).item(0), (centers[frame][2]).item(0), (centers[frame][2] + axis[2] * 1000).item(0)]

    ax.plot(axis_x, axis_y, axis_z, 'b')

def get_right_knee(file, start, end):
    vicon = Vicon.Vicon(file)
    markers = vicon.get_markers()

    markers.smart_sort()
    shank = markers.get_rigid_body("R_Tibia")
    thigh = markers.get_rigid_body("R_Femur")
    transforms = []
    error = []

    m1 = shank[0][start:end]
    m2 = shank[1][start:end]
    m3 = shank[2][start:end]
    m4 = shank[3][start:end]
    data = [m1, m2, m3, m4]

    core = Markers.calc_CoR(data)
    axis = Markers.calc_AoR(data)

    core = [[core[0][0]], [core[1][0]], [core[2][0]],[1.0]]
    core = np.array(core)
    vect = np.array([[0.0], [0.0], [0.0], [0.0]])
    max_error = 10000000000
    for frame in xrange(start, end):
        f = [shank[0][frame], shank[1][frame], shank[2][frame], shank[3][frame]]
        T, err = Markers.cloud_to_cloud(cloud, f)
        if err < max_error:
            max_error = err
            vect =  np.dot(np.linalg.pinv(T), core)
        error.append(err)
        transforms.append(T)

    #vect = vect/(end - start)
    centers = []
    for frame in xrange(len(shank[0])):
        f = [shank[0][frame], shank[1][frame], shank[2][frame], shank[3][frame]]
        T, err = Markers.cloud_to_cloud(cloud, f)
        point = np.dot(T, vect)[0:3]
        _thigh = Markers.calc_mass_vect([thigh[0][frame],
                                         thigh[1][frame],
                                         thigh[2][frame],
                                         thigh[3][frame]])

        _shank = Markers.calc_mass_vect([shank[0][frame],
                                         shank[1][frame],
                                         shank[2][frame],
                                         shank[3][frame]])

        sol =  Markers.minimize_center([_thigh, _shank], axis=axis, initial=(point[0][0], point[1][0], point[2][0]))
        centers.append( sol.x )
        #centers.append(point)


    keys = markers._filtered_markers.keys()
    nfr = len(markers._filtered_markers[keys[0]])  # Number of frames
    x_total = []
    y_total = []
    z_total= []

    for frame in xrange(nfr):
        x = []
        y = []
        z = []
        for key in keys:
            point = markers._filtered_markers[key][frame]
            x += [point.x]
            y += [point.y]
            z += [point.z]
        x_total.append(x)
        y_total.append(y)
        z_total.append(z)


    fps = 100  # Frame per sec
    keys = markers._filtered_markers.keys()
    nfr = len(markers._filtered_markers[keys[0]])  # Number of frames
    print "sldfj ",  nfr
    ani = animation.FuncAnimation(fig,
                                  animate, nfr,
                                  fargs=(x_total, y_total, z_total, centers, axis),
                                  interval=100 / fps)

    plt.show()

    return centers


def get_joint_location(file, joint, start, end):

    vicon = Vicon.Vicon(file)
    markers = vicon.get_markers()

    markers.smart_sort()
    markers = markers.get_rigid_body(joint)

    m1 = markers[0][start:end]
    m2 = markers[1][start:end]
    m3 = markers[2][start:end]
    m4 = markers[3][start:end]
    data = [m1, m2, m3, m4]

    core = Markers.calc_CoR(data)
    axis = Markers.calc_AoR(data)

    centers = []
    for i in xrange(len(markers[0])):
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

    #right_hip = get_joint_location(file, "R_Femur", 260, 750) # 750
    #right_knee = get_joint_location(file, "R_Tibia", 900, 950) # 950
    right_knee = get_right_knee(file, 875, 930) # 950

    #right_ankle = get_joint_location(file, 1100, 1400)

    # left_hip = get_joint_location(file, "L_Femur", 1600, 2000)
    # left_knee = get_joint_location(file, "L_Tibia", 2200, 2300)
    # left_ankle = get_joint_location(file, "L_Foot", 2500, 2600)

    # joints = [right_knee]
    #
    # data = Vicon.Vicon(file)
    # markers = data.get_markers()
    # markers.smart_sort()
    # markers.play(joints)
