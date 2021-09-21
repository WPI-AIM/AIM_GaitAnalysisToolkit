import GMM
import pyquaternion as pyq


class QGMM(GMM.GMM):



    def __init__(self, nb_states, nb_dim=3, reg=[1e-8]):
        super().__init__(nb_states, nb_dim, reg)

    def init_params(self, data):
        super().init_params(data)

    def train(self, data, maxiter=2000):
        return super().train(data, maxiter)

    def kmeansclustering(self, data):
        return super().kmeansclustering(data)

    def em(self, data, maxiter=2000):
        return super().em(data, maxiter)

    def displacement(self, p1, p2):
        '''
        get the displacement between two points
        :param p1: pyquaternion
        :param p2: pyquaternion
        :return:
        '''
        return pyq.Quaternion.distance(p1, p2.conjugate())



    def average(self, data, id):
        return  averageQuaternions(data[:, id])



    @staticmethod
    def averageQuaternions(q_list, w=[]):
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
        # Number of quaternions to average
        M = q_list.shape[0]
        A = np.zeros(shape=(4,4))
        W = np.ones(M)
        weightSum = 0
        if w:
            print(M == len(w))
            assert M == len(w),"len(quaternions) != len(weights)"
            W = np.array(w)


        for i in range(0,M):
            q = q_list[i].q
            # multiply q with its transposed version q' and add A
            A = W[i]*np.outer(q,q) + A
            weightSum += W[i]

        # scale
        A = (1.0/weightSum)*A
        # compute eigenvalues and -vectors
        eigenValues, eigenVectors = np.linalg.eig(A)
        # Sort by largest eigenvalue
        eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]
        # return the real part of the largest eigenvector (has only real part)
        return Quaternion(np.real(eigenVectors[:,0].A1))