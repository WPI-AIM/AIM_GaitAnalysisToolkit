

from Vicon import Vicon
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import matplotlib.pyplot as plt
import numpy as np

file = "/media/nathaniel/Data01/BioMechanocle_knee/Subject00/12162019/Subject00_Loaded_Bend_00.csv"
#file = "/media/nathaniel/Data01/stairclimbing_data/CSVs/subject_00/subject_00 walk_00.csv"
vicon = Vicon.Vicon(file)
markers = vicon.get_markers()
markers.smart_sort()
knee = markers.get_marker("RKNE")
inner_knee = markers.get_marker("R_Knee_Inner")
tibia = markers.get_marker("RTIBA")

virtual = []
tib = []
dist = []
for i in xrange(len(tibia)):

    virt = (0.5*knee[i] + 0.5*inner_knee[i]).toarray()
    ti = tibia[i].toarray()
    dist.append( np.sqrt( np.sum(np.power(virt - ti, 2)) ) )





fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.scatter(np.array(virtual)[:,0], np.array(virtual)[:,1], np.array(virtual)[:,2], marker='o')
#ax.scatter(np.array(tib)[:,0], np.array(tib)[:,1], np.array(tib)[:,2], marker='^')
plt.plot(dist)
plt.show()




