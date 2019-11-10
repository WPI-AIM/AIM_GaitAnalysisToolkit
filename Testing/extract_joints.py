from Vicon import Vicon
#data = Vicon.Vicon("/home/nathanielgoldfarb/gait_analysis_toolkit/testing_data/Range_of_Motion.csv")
data = Vicon.Vicon("/home/nathaniel/git/Gait_Analysis_Toolkit/testing_data/stairclimb03.csv")
model = data.get_model_output()
model.get_left_leg().hip.angle.x
fp = data.get_force_plate(1).get_forces()

