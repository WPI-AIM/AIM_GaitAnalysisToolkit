import numpy as np
from matplotlib.collections import PatchCollection
import numpy as np
import matplotlib
import numpy.polynomial.polynomial as poly
import numpy as np
import matplotlib.pyplot as plt



def calculate_imitation_metric_spatially(demos, imitation):
    M = len(demos)
    T = len(imitation)
    metric = 0.0

    for d in demos:
        metric += np.sum(np.sqrt(np.power( imitation.flatten() - d.flatten(),2)))

    return metric


def __get_gmm(Mu, Sigma, ax=None, index=1):
    nbDrawingSeg = 35
    t = np.linspace(-np.pi, np.pi, nbDrawingSeg)
    X = []
    nb_state = len(Mu[0])
    patches = []

    for i in range(nb_state):
        w, v = np.linalg.eig(Sigma[i])
        R = np.real(v.dot(np.lib.scimath.sqrt(np.diag(w))))
        x = R.dot(np.array([np.cos(t), np.sin(t)])) + np.matlib.repmat(Mu[:, i].reshape((-1, 1)), 1, nbDrawingSeg)
        x = x.transpose().tolist()
        patches.append(Polygon(x, edgecolor='r'))
        ax.plot(Mu[0, i], Mu[1, i], 'm*', linewidth=10)

    p = PatchCollection(patches, edgecolor='k', color='green', alpha=0.8)
    ax.add_collection(p)

    return p


def plot_gmm(runner,num_demo, ax,index=1):
    font = {'family': 'normal',
            'weight': 'bold',
            'size': 30}



    sIn = runner.get_sIn()
    tau = runner.get_tau()
    l = runner.get_length()
    motion = runner.get_motion()
    goals = runner.get_goals()
    mu = runner.get_mu()
    sigma = runner.get_sigma()
    currF = runner.get_expData()
    # plot the forcing functions
    dementions = len(tau)-1
    sigmas = []

    for j in range(len(currF)):
        for i in range(num_demo):
            ax[index+j].plot(sIn, tau[1+j, i * l: (i + 1) * l].tolist() + goals[j][i], color="b")
            ax[index+j].plot(sIn, currF[j].tolist(), color="y", linewidth=5)

    #

    for i in range(len(currF)):
        print(i)
        current_sigma = []
        mat_index = i+1
        for mat in sigma:
            current_sigma.append([[mat[0,0], mat[0,mat_index]], [ mat[0,mat_index], mat[mat_index,mat_index]]])

        current_sigma = np.array(current_sigma)
        p = __get_gmm(Mu=np.array([mu[0,:],  mu[1+i,:] ]), Sigma=current_sigma, ax=ax[index+i])