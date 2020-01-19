


from Vicon import Vicon
import numpy as np
import matplotlib.pyplot as plt
from Trajectories import center_of_rotation
from lib.Exoskeleton.Robot import core
from Vicon import Vicon
from Vicon import Markers
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import Axes3D

if __name__ == "__main__":
    file = "/media/nathaniel/Data01/BioMechanocle_knee/Subject01/12162019/Subject01_Unloaded_Bend_L02.csv"
    #file = "/media/nathaniel/Data01/stairclimbing_data/CSVs/subject_00/subject_00 walk_01.csv"

    vicon = Vicon.Vicon(file)
    markers = vicon.get_markers()
    markers.smart_sort()
    print markers.get_marker("LASI")
