import numpy as np
from math import sin as s
from math import cos as c
import math
from lib.Exoskeleton.Robot import core
import numpy.linalg as lin
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from Trajectories import rigid_marker

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')



import numpy as np
from Vicon import Markers
from Utilities import Mean_filter

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def make_frame(markers):
    origin = np.array([markers[0].x, markers[0].y, markers[0].z])
    x_axis = np.array([markers[1].x, markers[1].y, markers[1].z])
    y_axis = np.array([markers[2].x, markers[2].y, markers[2].z])
    xo = unit_vector(origin - x_axis)
    yo = unit_vector(origin - y_axis)
    zo = unit_vector(np.cross(xo,yo))
    xo = np.pad(xo, (0, 1), 'constant')
    yo = np.pad(yo, (0, 1), 'constant')
    zo = np.pad(zo, (0, 1), 'constant')
    p = np.pad(origin, (0, 1), 'constant')
    p[-1] = 1
    F = np.column_stack((xo,yo,zo,p))
    return F

def leastsq_method(markers, offset=0):

    centers = []
    cor_filter = Mean_filter.Mean_Filter(10)

    for frame in xrange(offset, len(markers[0]) - offset):
        m1 = markers[0][frame - offset:frame + offset + 1]
        m2 = markers[1][frame - offset:frame + offset + 1]
        m3 = markers[2][frame - offset:frame + offset + 1]
        m4 = markers[3][frame - offset:frame + offset + 1]
        data = [m1, m2, m3, m4]
        core = cor_filter.update(Markers.calc_CoR(data))

        centers.append(core)

    return centers


def rotation_method2(markers,offset=0):

    for frame in xrange(offset, len(markers[0]) - offset):
        m1 = markers[0][frame:frame + 2]
        m2 = markers[1][frame:frame + 2]
        m3 = markers[2][frame:frame + 2]
        m4 = markers[3][frame:frame + 2]
        data = [m1, m2, m3, m4]
        R, t = Markers.get_transformation(data)


def rotation_method(markers,offset=1):

    centers = []
    axises = []
    offset = 5
    for frame in xrange(offset, len(markers[0]) - offset):

        data = []
        m1 = markers[0][frame]
        m2 = markers[1][frame]
        m3 = markers[2][frame]
        m4 = markers[3][frame]
        data = [m1, m2, m3, m4]
        T_Sh1 = make_frame(data)
        
        m1 = markers[0][frame + offset]
        m2 = markers[1][frame + offset]
        m3 = markers[2][frame + offset]
        m4 = markers[3][frame + offset]

        data = [m1, m2, m3, m4]
        T_Sh2 = make_frame(data)

        T_Th = np.asarray([ [1.0, 0.0, 0.0, 0.0],
                            [0.0, 1.0, 0.0, 0.0],
                            [0.0, 0.0, 1.0, 0.0],
                            [0.0, 0.0, 0.0, 1.0]])

        T_TH_SH_1 = T_Sh1  # Markers.get_all_transformation_to_base(T_Th, T_Sh)[frame]
        T_TH_SH_2 = T_Sh2

        R1 = T_TH_SH_1[:3, :3]
        R2 = T_TH_SH_2[:3, :3]
        R1_2 = np.dot(np.transpose(R2), R1)

        # rp_1 = Markers.calc_mass_vect([markers[0][frame],
        #                                markers[1][frame],
        #                                markers[2][frame],
        #                                markers[3][frame]])
        #
        # rp_2 = Markers.calc_mass_vect([markers[0][frame + 1],
        #                                markers[1][frame + 1],
        #                                markers[2][frame + 1],
        #                                markers[3][frame + 1]])

        rp_1 = np.asarray([[markers[0][frame].x],[markers[0][frame].y],[markers[0][frame].z]])
        rp_2 = np.asarray([[markers[0][frame+offset].x], [markers[0][frame+offset].y], [markers[0][frame+offset].z]])
        rd_1 = np.asarray([[0.0], [0.0], [0.0]])
        rd_2 = np.asarray([[0.0], [0.0], [0.0]] )

        rdp1 = np.dot(T_Sh1[:3,:3], rd_1 - rp_1)
        rdp2 = np.dot(T_Sh2[:3,:3], rd_2 - rp_2)
        P = np.eye(3) - R1_2
        Q = rdp2 - np.dot(R1_2, rdp1)

        rc = np.dot(np.linalg.pinv(P), Q)

        axis, angle = Markers.R_to_axis_angle(T_TH_SH_1[0:3, 0:3])
        Rc = rp_1 + np.dot(np.transpose(T_Sh1[:3,:3]), rc.reshape((-1,1)))
        centers.append(Rc)
        print Rc
        axises.append(axis)

    return centers, axises

def sphere_method(markers, offset=10):

    centers = []
    axises = []
    CoM = []
    CoM_fixed = []
    # Get all the mass centers of the frame
    for ii in xrange(len(markers[0])):
        temp = Markers.calc_mass_vect([markers[0][ii],
                                       markers[1][ii],
                                       markers[2][ii],
                                       markers[3][ii]])
        CoM.append(np.asarray(temp))

    fixed = CoM[0]
    CoM_fixed.append(fixed)
    thresh = 0.0
    for center in CoM:
        dist = np.sqrt(np.sum(np.power(fixed-center,2)))
        if dist >= thresh:
            fixed = center
            CoM_fixed.append(fixed)

    for ii in xrange( len(CoM_fixed)-5):
        raduis, C = Markers.sphereFit(CoM_fixed[ii:ii+5])
        print "center ",  C
        C = np.row_stack((C,[1]))

        centers.append(C[0:3])

    return centers

def getD(x,y,z):
    return np.matrix( [[1, 0, 0, -x,], [0, 1, 0, -y,], [0, 0, 1, -z,], [0, 0, 0, 1]])

def getRx(A,B,C):
    V = np.sqrt( B**2 + C**2)
    return np.matrix([ [1, 0, 0, 0],  [0, C/V, -B/V, 0], [0, B/V, C/V, 0], [0, 0, 0, 1]])

def getRy(A,B,C):
    L = np.sqrt( A**2 + B ** 2 + C ** 2)
    V = np.sqrt(B ** 2 + C ** 2)
    return np.matrix([[V/L,0, -A/L, 0], [0, 1, 0, 0], [A/L, 0, V/L, 0], [0, 0, 0, 1]])

def getRz(theta):
    return np.matrix( [ [ c(theta), -s(theta),0 ,0 ], [ s(theta), c(theta), 0, 0 ], [ 0 ,0,1,0 ], [ 0,0,0,1 ]])


def getT(N, M, theta):
    A = M[0] - N[0]
    B = M[1] - N[1]
    C = M[2] - N[2]
    Rx = getRx(A,B,C)
    Ry = getRy(A,B,C)
    Rz = getRz(theta)
    D = getD(N[0], N[1], N[2])
    T = lin.inv(D)*lin.inv(Rx)*lin.inv(Ry)*Rz*Ry*Rx*D
    return T

N = (0,0,5)
M = (0,1,5)

P = np.matrix( [ [0,  8,   0,  4],
                 [4,  4,   4,  4],
                 [0,  0,   8,  4],
                 [1,   1,  1,  1]])

all_points = []
theta = math.radians(1)
all_points.append(P)
x = []
y = []
z = []

for i in xrange(0,60):

    x.append(P[0,0:])
    y.append(P[1,0:])
    z.append(P[2,0:])
    P = getT(N, M, theta)*P
    all_points.append(P)

ax.scatter(x,y,z)
m1 = []
m2 = []
m3 = []
m4 = []

for idx in range(0,60):

    P = all_points[idx][:,0]
    point = core.Point( P[0].item(0), P[1].item(0), P[2].item(0))
    m1.append(point)

    P = all_points[idx][:,1]
    point = core.Point( P[0].item(0), P[1].item(0), P[2].item(0))
    m2.append(point)

    P = all_points[idx][:,2]
    point = core.Point( P[0].item(0), P[1].item(0), P[2].item(0))
    m3.append(point)

    P = all_points[idx][:,3]
    point = core.Point( P[0].item(0), P[1].item(0), P[2].item(0))
    m4.append(point)

markers = [ m1, m2, m3, m4]
# center = Markers.calc_CoR(markers)
#centers = sphere_method(markers)
#centers = leastsq_method(markers,2)
centers = rotation_method(markers)
rotation_method2(markers)
x = []
y = []
z = []

for ii in xrange(len(centers)-1):
    x.append(centers[ii][0])
    y.append(centers[ii][1])
    z.append(centers[ii][2])

ax.scatter(x,y,z)
plt.show()
