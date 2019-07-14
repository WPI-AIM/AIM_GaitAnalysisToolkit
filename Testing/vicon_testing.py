from Vicon import Vicon

data = Vicon.Vicon("/home/nathaniel/git/Gait_Analysis_Toolkit/testing_data/Walking01.csv")
print data.get_model_output()
