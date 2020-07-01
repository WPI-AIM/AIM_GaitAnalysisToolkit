#!/usr/bin/env python3

# //==============================================================================
# /*
#     Software License Agreement (BSD License)
#     Copyright (c) 2020, WPIGaitAnalysisToolKit
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
import sys
import os

import matplotlib.pyplot as plt
from GaitCore.Core import utilities
from scipy import signal
import numpy as np
from ..Session import ViconGaitingTrial
import Vicon
import os


"""
This shows how to use the gaiting tools, it seperates the joint trajectores and 
plots the joint angle and moments

"""


def compare_walking_angles(files, list_of_index):
    """
    Plots the joint angle during walking
    :param files: sting of file name
    :param list_of_index: which traj to use
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Angles', fontsize=20)
    resample = 100000

    # find resample length to use
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.vicon.get_model_output().get_right_leg().hip.angle
        sample = len(joints["Rhip"][i].angle.data)
        resample = min(resample, sample)

    # grab all the trajs and resample
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip = signal.resample( joints["Rhip"][i].angle.data, resample)
        knee = signal.resample(joints["Rknee"][i].angle.data, resample)
        ankle = signal.resample(joints["Rankle"][i].angle.data, resample)
        ax1.plot(hip)
        ax2.plot(knee)
        ax3.plot(ankle)


    font_size = 25
    ax1.set_ylabel("Degrees", fontsize=font_size)
    ax2.set_ylabel("Degrees", fontsize=font_size)
    ax3.set_ylabel("Degrees", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()


def compare_walking_moments(files, list_of_index):
    """
      Plots the joint angle during walking
      :param files: sting of file name
      :param list_of_index: which traj to use
      """
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking Joint Angles', fontsize=20)
    resample = 100000

    # find resample length to use
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.vicon.get_model_output().get_right_leg().hip.angle
        sample = len(joints)
        resample = min(resample, sample)

    # grab all the trajs and resample
    for file, i in zip(files, list_of_index):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        joints = trial.get_joint_trajectories()
        hip = signal.resample(trial.vicon.get_model_output().get_right_leg().hip.moment, resample)
        knee = signal.resample(trial.vicon.get_model_output().get_right_leg().knee.moment, resample)
        ankle = signal.resample(trial.vicon.get_model_output().get_right_leg().ankle.moment, resample)
        ax1.plot(utilities.smooth(hip, 10))
        ax2.plot(utilities.smooth(knee, 10))
        ax3.plot( utilities.smooth(ankle,10))

    font_size = 25
    ax1.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax2.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax3.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)


    plt.show()


def compare_stair_angles(files, side, legend):

    resample = 100000
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Stair Joint Angles', fontsize=20)

    indiecs = {}
    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(file)
        rn = trial.get_stair_ranges(s)
        indiecs[file] = rn
        resample = min(resample, rn[1] - rn[0])

    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        if s == "R":
            joints = trial.vicon.get_model_output().right_leg()
        else:
            joints = trial.vicon.get_model_output().left_leg()
        rn = indiecs[file]
        hip = signal.resample(joints.hip.angle.x[rn[0]:rn[1]], resample)
        knee = signal.resample(joints.knee.angle.x[rn[0]:rn[1]], resample)
        ankle = signal.resample(joints.ankle.angle.x[rn[0]:rn[1]], resample)
        ax1.plot(hip)
        ax2.plot(knee)
        ax3.plot(ankle)

    plt.legend(legend)
    font_size = 25
    ax1.set_ylabel("Degrees", fontsize=font_size)
    ax2.set_ylabel("Degrees", fontsize=font_size)
    ax3.set_ylabel("Degrees", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()


def compare_stair_moments(files, side, legend):
    resample = 100000
    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Stair Joint Moments', fontsize=20)

    resample = 1000000000

    indiecs = {}
    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(file)
        rn = trial.get_stair_ranges(s)
        indiecs[file] = rn
        resample = min(resample, rn[1] - rn[0])

    for file, s in zip(files, side):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        if s == "R":
            joints = trial.vicon.get_model_output().right_leg()
        else:
            joints = trial.vicon.get_model_output().left_leg()
        rn = indiecs[file]
        hip = signal.resample(joints.hip.angle.x[rn[0]:rn[1]], resample)
        knee = signal.resample(joints.knee.angle.x[rn[0]:rn[1]], resample)
        ankle = signal.resample(joints.ankle.angle.x[rn[0]:rn[1]], resample)
        ax1.plot(utilities.smooth(hip,10))
        ax2.plot( utilities.smooth(knee,10))
        ax3.plot(utilities.smooth(ankle,10))

    #plt.legend(legend)
    font_size = 25
    ax1.set_ylabel("Nmm/kg", fontsize=font_size)
    ax2.set_ylabel("Nmm/kg", fontsize=font_size)
    ax3.set_ylabel("Nmm/Kg", fontsize=font_size)
    ax1.set_title("Hip", fontsize=font_size)
    ax2.set_title("Knee", fontsize=font_size)
    ax3.set_title("Ankle", fontsize=font_size)
    plt.xlabel("Gait %", fontsize=font_size)

    plt.show()




if __name__ == "__main__":

    script_dir = os.path.dirname(__file__)  # <-- absolute dir the script is in

    compare_walking_angles(
        [os.path.join(script_dir, "ExampleData/subject_00 walk_00.csv"),
         os.path.join(script_dir, "ExampleData/subject_01 walk_00.csv")],
        [1, 8])

    compare_walking_moments(
        [os.path.join(script_dir, "ExampleData/subject_00 walk_00.csv"),
         os.path.join(script_dir, "ExampleData/subject_01 walk_00.csv")],
        [1, 8])


