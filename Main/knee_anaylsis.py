import sys
import matplotlib.pyplot as plt
import numpy as np
from lib.dmp_experiments.Python import train_dmp, DMP_runner
from Session import Trial
from Vicon import Vicon

def plot_data():
    trial = Trial.Trial(vicon_file="/home/nathanielgoldfarb/stairclimbing/subject_00/11_07_2019/subject_00 walk_00.csv")
    joints = trial.get_joint_trajectories()

    fig, (ax1, ax2, ax3) = plt.subplots(3, sharex=True)
    fig.suptitle('Walking data')

    for i in [1,3]:  #xrange(2,len(joints["Rknee"])):
        ax1.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].angle.data )
        ax2.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].moment.data)
        ax3.plot(joints["Rknee"][i].moment.time, joints["Rknee"][i].power.data)

    plt.show()


plot_data()


if __name__ == "__main__":
    plot_data()