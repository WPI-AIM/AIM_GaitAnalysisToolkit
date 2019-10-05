
from Vicon import Vicon
from Trajectories import rigid_marker
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.animation as animation
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, c, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), color=c,*args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)




ax.set_autoscale_on(False)
data = Vicon.Vicon("/home/nathaniel/gait_analysis_toolkit/testing_data/ben_leg_bend.csv")
markers = data.get_markers()
markers.smart_sort()
markers.auto_make_frames()
shank_markers = markers.get_rigid_body("ben:RightShank")[0:300]

def animate(frame):

    x = []
    y = []
    z = []

    T_RTh = markers.get_frame("ben:RightThigh")[frame]
    T_RSh = markers.get_frame("ben:RightShank")[frame]
    T_H = markers.get_frame("ben:hip")[frame]


    x = [T_H[:,-1][0], T_RTh[:,-1][0], T_RSh[:,-1][0]]
    y = [T_H[:,-1][1], T_RTh[:,-1][1], T_RSh[:,-1][1]]
    z = [T_H[:,-1][2], T_RTh[:,-1][2], T_RSh[:,-1][2]]
    arrow_prop_dict = dict(mutation_scale=5, arrowstyle='->', shrinkA=0, shrinkB=0)

    ax.clear()
    for coor in [T_H, T_RTh, T_RSh]:
        og = coor[:,-1]
        axis_x = coor[:, 0]
        axis_y = coor[:, 1]
        axis_z = coor[:, 2]
        # size = 1
        # a = Arrow3D([ og[0] + axis_x[0], og[0]], [ og[1] + axis_x[1], og[1]], [ og[2] + axis_x[2], og[2]], 'r',  **arrow_prop_dict)
        # ax.add_artist(a)
        # a = Arrow3D([ og[0] + axis_y[0], og[0]], [ og[1] + axis_y[1], og[1]], [ og[2] + axis_y[2], og[2]], 'g', **arrow_prop_dict)
        # ax.add_artist(a)
        # a = Arrow3D([ og[0] + axis_z[0], og[0]], [ og[1] + axis_z[1], og[1]], [ og[2] + axis_z[2], og[2]], 'b', **arrow_prop_dict)
        # ax.add_artist(a)

        size =1
        a = Arrow3D([og[0], og[0] + axis_x[0] + size ], [og[1], og[1] + axis_x[1] +size], [og[2], og[2] + axis_x[2] + size] , 'r',
                    **arrow_prop_dict)
        ax.add_artist(a)
        a = Arrow3D([og[0], og[0] + axis_y[0] + size], [og[1], og[1] + axis_y[1] + size ], [og[2], og[2] + axis_y[2] + size], 'g',
                    **arrow_prop_dict)
        ax.add_artist(a)
        a = Arrow3D([og[0], og[0] + axis_z[0] + size], [og[1], og[1] + axis_z[1] + size ], [og[2], og[2] + axis_z[2] + size ], 'b',
                    **arrow_prop_dict)
        ax.add_artist(a)

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.axis([-500, 500, -2000, 3000])
    ax.scatter(x, y, z, c='r', marker='o')




ani = animation.FuncAnimation(fig, animate, interval=0.1)
plt.show()