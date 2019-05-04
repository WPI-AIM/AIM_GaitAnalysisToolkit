import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import find_peaks_cwt
import pandas


def sperate_joints(file_path):

    gait_data = {}
    column = []
    name = "RHipAngles"

    gait_data = pd.read_csv(file_path)
    hip = np.array(gait_data["LHipAngles"] ).flatten()
    start = np.argmax( np.array(gait_data["LHipAngles"]) > 0 )
    dH = np.gradient(hip)

    max_peakind = np.diff(np.sign(np.diff(hip))).flatten() #the one liner
    max_peakind = np.pad(max_peakind, (1, 1), 'constant', constant_values=(0, 0))
    max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]

    joints = {}

    names = ["LHipAngles","LKneeAngles","LAbsAnkleAngle","RHipAngles", "RKneeAngles", "RAbsAnkleAngle"]

    for name in names:
        joints[name] = []
        for start in xrange(2,len(max_peakind)-2):
            joints[name].append(np.array(gait_data[name][max_peakind[start]:max_peakind[start+1]]))

    return joints


