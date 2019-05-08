import csv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.signal import find_peaks_cwt
import pandas

def sperate_joints2(file_path):
    gait_data = {}
    column = []

    gait_data = pd.read_csv(file_path)
    knee = np.array(gait_data["LKneeAngles"]).flatten()
    max_peakind = np.diff(np.sign(np.diff(knee))).flatten()  # the one liner
    max_peakind = np.pad(max_peakind, (1, 1), 'constant', constant_values=(0, 0))
    max_peakind = [index for index, value in enumerate(max_peakind) if value == -2 and knee[index] > 40.0 ]
    searching = True
    last_index = 0
    indecies = []
    starting_index = 0
    final_index = 0
    for peak_index in max_peakind:

        #get the starting index
        last_angle = -1000
        for index in xrange(starting_index, peak_index):
            angle = knee[index]
            if angle > 0 and angle > last_angle:
                starting_index = index
                break
            else:
                last_angle = angle

        searching = True
        index = peak_index
        while searching:
            if knee[index] < 0:
                final_index = index-1
                searching = False
            elif knee[index] > knee[index-1]:
                searching = False
                final_index = index
            index = index + 1

        indecies.append([starting_index, final_index+1 ])
        starting_index = final_index-10
    names = ["LHipAngles", "LKneeAngles", "LAbsAnkleAngle", "RHipAngles", "RKneeAngles", "RAbsAnkleAngle"]

    joints = {}
    for name in names:
        joints[name] = []
        for index in indecies:
            starting_index = index[0]
            final_index = index[1]
            joints[name].append(np.array(gait_data[name][starting_index:final_index]))

    return joints




def sperate_joints(file_path):

    gait_data = {}
    column = []
    name = "RHipAngles"

    gait_data = pd.read_csv(file_path)
    hip = np.array(gait_data["LHipAngles"] ).flatten()
    start = np.argmax( np.array(gait_data["LHipAngles"]) > 0 )
    dH = np.gradient(hip)

    max_peakind = np.diff(np.sign(np.diff(hip))).flatten() #the one liner

    max_peakind = np.pad(max_peakind, (1, 10), 'constant', constant_values=(0, 0))
    max_peakind = [index for index, value in enumerate(max_peakind) if value == -2]

    joints = {}

    names = ["LHipAngles","LKneeAngles","LAbsAnkleAngle","RHipAngles", "RKneeAngles", "RAbsAnkleAngle"]
    offsets = []

    # for start in xrange(2, len(max_peakind) - 2):
    #     error = 10000000
    #     offset = 0
    #     starting_value = gait_data["LHipAngles"][max_peakind[start]]
    #     for ii in xrange(-10, 20):
    #         temp_error = abs(starting_value - gait_data["LHipAngles"][max_peakind[start+1] + ii])
    #         if temp_error < error:
    #             error = temp_error
    #             offset = ii
    #     offsets.append(offset)


    for start in xrange(2, len(max_peakind) - 2):
        error = 10000000
        offset = 0
        for ii in xrange(0, 25):
            temp_error = gait_data["LKneeAngles"][max_peakind[start+1] + ii]
            if temp_error < error:
                error = temp_error
                offset = ii
        offsets.append(offset)


    for name in names:
        joints[name] = []
        for ii, start in enumerate(xrange(2, len(max_peakind) - 2)):
            joints[name].append(np.array(gait_data[name][max_peakind[start]:max_peakind[start+1]+offsets[ii]]))

    return joints



