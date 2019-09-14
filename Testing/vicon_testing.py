from Vicon import Vicon
from Trajectories import rigid_marker
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ridgid_markers.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
frame = markers.get_rigid_body("LowerLegs_Healthy:RightThigh")

#print frame[2]
rigid_marker.find_CoR(frame)

