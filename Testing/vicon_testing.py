import matplotlib.pyplot as plt

from EMG import EMG_Toolkit
from Session import Trial

vicon_file = "/home/nathaniel/git/Gait_Analysis_Toolkit/Utilities/Walking01.csv"
config_file = "/home/nathaniel/git/exoserver/Config/sensor_list.yaml"
exo_file = "/home/nathaniel/git/exoserver/Main/subject_1234_trial_1.csv"
trial = Trial.Trial(vicon_file, config_file, exo_file)
joints = trial.seperate_joint_trajectories()
plate = trial.seperate_force_plates()
left, right = trial.seperate_CoP()
emg = trial.seperate_emg()
data1 = EMG_Toolkit.remove_mean(emg[1][0].data)
data1 = EMG_Toolkit.window_rms(data1, 300)
plt.plot(data1)
plt.ylabel('some numbers')
plt.show()
