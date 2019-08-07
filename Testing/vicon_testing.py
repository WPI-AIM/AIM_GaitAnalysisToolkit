import matplotlib.pyplot as plt

from EMG import EMG_Toolkit
from Session import Trial

vicon_file = "/media/nathaniel/Data/LowerLimb_HealthyGait/Subject02/walking/Walking03.csv"
config_file = "/home/nathaniel/git/exoserver/Config/sensor_list.yaml"
exo_file = "/home/nathaniel/git/exoserver/Main/subject_1234_trial_1.csv"
trial = Trial.Trial(vicon_file, config_file, exo_file)
joints = trial.seperate_joint_trajectories()
plate = trial.seperate_force_plates()
cop = trial.seperate_CoP()
emgs = trial.seperate_emg()

#joint = joints["LKneeAngles"]
for joint in joints["LKneeAngles"]:
    plt.plot(joint.time, joint.data )



# for emg in emgs[2]:
#     data = EMG_Toolkit.remove_mean(emg.data)
#     data = EMG_Toolkit.butterworth(data)
#     data = EMG_Toolkit.rectify(data)
#     data = EMG_Toolkit.low_pass(data)
#     plt.plot(data)


plt.ylabel('some numbers')
plt.show()
