import numpy as np
import scipy as sp


def spline(x, Y, xx, kind='cubic'):
    ''' Attempts to imitate the matlab version of spline'''
    # from scipy.interpolate import interp1d
    if Y.ndim == 1:
        return sp.interpolate.interp1d(x, Y, kind=kind)(xx)
    F = [sp.interpolate.interp1d(x, Y[:, i]) for i in range(Y.shape[1])]
    return np.vstack([f(xx) for f in F])
