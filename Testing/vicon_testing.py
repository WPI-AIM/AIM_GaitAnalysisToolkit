from Vicon import Vicon
from Trajectories import rigid_marker
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ridgid_markers.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
frame = markers.get_rigid_body("LowerLegs_Healthy:RightFoot")
rigid_marker.find_CoR(frame)

