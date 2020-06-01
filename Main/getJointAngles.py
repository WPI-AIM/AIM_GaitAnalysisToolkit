import sys
import matplotlib.pyplot as plt
import lib.GaitCore.Core.utilities as utilities
from scipy import signal
import numpy as np
from Session import ViconGaitingTrial
from lib.Vicon import Vicon
import os


def plot_joint_angles(files, indecies, lables):

    for file, i in zip(files, indecies):
        trial = ViconGaitingTrial.ViconGaitingTrial(vicon_file=file)
        trial.create_index_seperators()
        body = trial.get_joint_trajectories()






if __name__ == "__main__":
    plot_joint_angles(["/home/nathaniel/AIM_GaitData/Gaiting_stairs/subject_10/subject_10 walk_02.csv"],
                      [2],
                      ["dfa"])

