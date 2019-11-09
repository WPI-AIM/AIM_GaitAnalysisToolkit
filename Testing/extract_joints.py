from Vicon import Vicon
#data = Vicon.Vicon("/home/nathanielgoldfarb/gait_analysis_toolkit/testing_data/Range_of_Motion.csv")
data = Vicon.Vicon("/home/nathanielgoldfarb/Documents/gait_analysis_toolkit/testing_data/stairclimb03.csv")
markers = data.get_markers()
markers.smart_sort()

model = data.get_model_output()
print model
