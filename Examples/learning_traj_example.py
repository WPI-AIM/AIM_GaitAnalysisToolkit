#!/usr/bin/env python
# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2020, WPIGaitAnaklsisToolKit
#     (www.aimlab.wpi.edu)

#     All rights reserved.

#     Redistribution and use in source and binary forms, with or without
#     modification, are permitted provided that the following conditions
#     are met:

#     * Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.

#     * Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.

#     * Neither the name of authors nor the names of its contributors may
#     be used to endorse or promote products derived from this software
#     without specific prior written permission.

#     THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
#     "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
#     LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
#     FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
#     COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
#     INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
#     BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
#     LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#     CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
#     LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
#     ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
#     POSSIBILITY OF SUCH DAMAGE.

#     \author    <http://www.aimlab.wpi.edu>
#     \author    <nagoldfarb@wpi.edu>
#     \author    Nathaniel Goldfarb
#     \version   0.1
# */
# //==============================================================================

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat # loading data from matlab
import os
import lib.pbdlib.pbdlib as pbd
import numpy.matlib
import lib.pbdlib.pbdlib.plot
#from pbdlib.utils.jupyter_utils import *


def getTraj(demos, samples):

    nbData = 200  # Length of each trajectory
    dt = 0.01
    kp = 50.0
    kv = (2.0 * kp) ** 0.5
    alpha = 1.0
    x_ = None
    dx_ = None
    ddx_ = None
    sIn = []
    taux = []
    tauy = []

    sIn.append(1.0)  # Initialization of decay term
    for t in xrange(1, nbData):
        sIn.append(sIn[t - 1] - alpha * sIn[t - 1] * dt)  # Update of decay term (ds/dt=-alpha s) )

    goal = demos[0][-1]

    for n in xrange(samples):
        demo = demos[n]
        size = demo.shape[0]
        x = pbd.functions.spline(np.arange(1, size + 1), demo, np.linspace(1, size, nbData))
        dx = np.divide(np.diff(x, 1), np.power(dt, 1.0))
        dx = np.vstack((np.append([0.0], dx[0]), np.append([0.0], dx[1])))
        ddx = np.divide(np.diff(x, 2), np.power(dt, 2))
        ddx = np.vstack((np.append([0.0, 0.0], ddx[0]), np.append([0.0, 0.0], ddx[1])))
        goals = np.matlib.repmat(goal, nbData, 1)
        tau_ = ddx - (kp * (goals.transpose() - x)) / sIn + (kv * dx) / sIn
        #demos_ = [[s, x_, y_ ] for s, x_, y_ in zip(sIn, tau_[0].tolist(), tau_[1].tolist() )]
        print x.tolist()
        if x_ is not None:
            x_ = x_ + x.tolist()
            dx_ = dx_ + dx.tolist()
            ddx_ = ddx_ + ddx.tolist()
        else:
            x_ = x.tolist()
            dx_ = dx.tolist()
            ddx_ = ddx.tolist()

        taux = taux + tau_[0].tolist()
        tauy = tauy + tau_[1].tolist()
    tau = np.vstack((sIn * samples, taux, tauy))
    motion = np.vstack((x_, dx_, ddx_))
    print "motion ", motion
    return tau, motion, sIn
    # return tau
    #     tau.append(np.array(demos_))

    #return tau



if __name__ == "__main__":
    np.set_printoptions(precision=3)
    nb_states = 5  # choose the number of states in HMM or clusters in GMM
    samples = 4
    datapath = os.path.dirname(pbd.__file__) + '/data/2Dletters/'
    data_in = loadmat(datapath + '%s.mat' % "G")
    demos = [d['pos'][0][0].T for d in data_in['demos'][0]] # cleaning matlab data
    tau, motion, sIn = getTraj(demos, samples=samples)

    gmm = pbd.GMM_Prime(nb_states=nb_states, nb_dim=3)

    gmm.init_params_kmeans(tau)

    gmm.em(tau, no_init=True)
    expData, expSigma, H = gmm.gmr( sIn, [0], [1,2])

    x = motion[0:2, 0].reshape((-1,1))
    dx = np.array([[0.0],[0.0]])
    L = np.append(np.eye(2) * 50.0, np.eye(2) * 10.0,1)
    xTar = np.array([ [ 0.1093],[ -0.5052]  ])

    my_x = []
    my_y = []
    for i in xrange(200):
        x_ = np.append( xTar-x ,  -dx ).reshape(((-1,1)))
        ddx = L.dot(x_) + (expData[:,i]*sIn[i]).reshape((-1,1))
        dx = dx + ddx * 0.01
        x = x + dx * 0.01
        my_x.append(x[0])
        my_y.append(x[1])


    plt.plot(my_x, my_y)
    plt.show()
        #print dxx







