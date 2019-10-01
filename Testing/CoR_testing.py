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
P = np.matrix( [ [0, 0, 0, 0], [0,0, -5,-5 ], [-5,-30,-15,-15], [1, 1, 1, 1 ] ] )

all_points = []
theta = math.radians(1)
all_points.append(P)
x = []
y = []
z = []
B = np.matrix( [ [0.05, 0.05, 0.05, 0.05], [0.05, 0.05, 0.05, 0.05 ], [0.05,0.05,0.05,0.05], [1, 1, 1, 1 ] ] )

for i in xrange(0,60):
    x.append(P[0,0:])

    y.append(P[1,0:])
    z.append(P[2,0:])
    P = getT(N, M, theta)*P + B
    all_points.append(P)
ax.scatter(x,y,z)


for ii in xrange(5,7):
    m1 = []
    m2 = []
    m3 = []
    m4 = []

    for idx in range(ii,ii+2):
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
    cor = rigid_marker.find_CoR(markers) + np.array((0.05,0.05,0.05))
    axis = rigid_marker.find_AoR(markers)
    axis_x = [(cor[0] - axis[0] * 100).item(0), (cor[0]).item(0), (cor[0] + axis[0] * 100).item(0)]
    axis_y = [(cor[1] - axis[1] * 100).item(0), (cor[1]).item(0), (cor[1] + axis[1] * 100).item(0)]
    axis_z = [(cor[2] - axis[2] * 100).item(0), cor[2].item(0), (cor[2] + axis[2] * 100).item(0)]
    ax.plot(axis_x, axis_y, axis_z, 'b')
plt.show()
