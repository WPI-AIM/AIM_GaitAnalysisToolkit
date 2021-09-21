# This file implements correct quaternion averaging.
#
# This method is computationally expensive compared to naive mean averaging.
# If only low accuracy is required (or the quaternions have similar orientations),
# then quaternion averaging can possibly be done through simply averaging the
# components.
#
# Based on:
#
# Markley, F. Landis, Yang Cheng, John Lucas Crassidis, and Yaakov Oshman.
# "Averaging quaternions." Journal of Guidance, Control, and Dynamics 30,
# no. 4 (2007): 1193-1197.
# Link: https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20070017872.pdf
#
# Code based on:
#
# Tolga Birdal. "averaging_quaternions" Matlab code.
# http://jp.mathworks.com/matlabcentral/fileexchange/40098-tolgabirdal-averaging-quaternions
#
# Comparison between different methods of averaging:
#
# Claus Gramkow. "On Averaging Rotations"
# Journal of Mathematical Imaging and Vision 15: 7–16, 2001, Kluwer Academic Publishers.
# https://pdfs.semanticscholar.org/ebef/1acdcc428e3ccada1a047f11f02895be148d.pdf
#
# Side note: In computer graphics, averaging or blending of two quaternions is often done through
# spherical linear interploation (slerp). Even though it's often used it might not be the best
# way to do things, as described in this post:
#
# Jonathan Blow.
# "Understanding Slerp, Then Not Using It", February 2004
# http://number-none.com/product/Understanding%20Slerp,%20Then%20Not%20Using%20It/
#

import numpy
import numpy as np
import numpy.matlib as npm
from pyquaternion import Quaternion

# Q is a Nx4 numpy matrix and contains the quaternions to average in the rows.
# The quaternions are arranged as (w,x,y,z), with w being the scalar
# The result will be the average quaternion of the input. Note that the signs
# of the output quaternion can be reversed, since q and -q describe the same orientation
def averageQuaternions(q_list, w=[]):
    # Number of quaternions to average
    M = q_list.shape[0]
    A = npm.zeros(shape=(4,4))
    A2 = np.zeros(shape=(4,4))
    W = np.ones(M)
    weightSum = 0
    if w:
        print(M == len(w))
        assert M == len(w),"len(quaternions) != len(weights)"
        W = np.array(w)


    for i in range(0,M):
        q = q_list[i].q
        # multiply q with its transposed version q' and add A
        A = W[i]*numpy.outer(q,q) + A
        A2 = W[i]*numpy.outer(q,q) + A2
        weightSum += W[i]

    # scale
    A = (1.0/weightSum)*A
    A2 = (1.0/weightSum)*A2
    # compute eigenvalues and -vectors
    eigenValues, eigenVectors = numpy.linalg.eig(A)
    eigenValues2, eigenVectors2 = numpy.linalg.eig(A2)
    # Sort by largest eigenvalue
    eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
    # return the real part of the largest eigenvector (has only real part)
    return Quaternion(numpy.real(eigenVectors[:,0].A1))



if __name__ == '__main__':

    Q = []
    w = []
    for i in range(10):
        myq =  Quaternion.random()
        Q.append(myq)
        w.append(1)
    Q = np.array(Q)


    print(averageQuaternions(Q,w))