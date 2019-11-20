from lib.Exoskeleton.Robot import core
import numpy as np
from scipy.optimize import minimize
import math
from numpy import *
from math import sqrt
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.animation as animation


class Markers(object):
    """
    Creates an object to hold marker values
    """

    def __init__(self, marker_dict):
        """

        :param marker_dict: dict of markers
        """
        self._data_dict = marker_dict
        self._raw_markers = {}
        self._rigid_body = {}
        self._marker_names = []
        self._frames = {}
        self._filter_window = 10
        self._filtered_markers = {}

    @property
    def marker_names(self):
        """
        :return marker names
        :return:
        """
        return self._marker_names

    @property
    def filter_window(self):
        return self._filter_window

    @filter_window.setter
    def filter_window(self, value):
        self._filter_window = value

    @property
    def filtered_markers(self):
        return self._filtered_markers

    @filtered_markers.setter
    def filtered_markers(self, value):
        self._filtered_markers = value

    @property
    def rigid_body(self):
        return self._rigid_body

    @filtered_markers.setter
    def rigid_body(self, value):
        self._rigid_body = value

    def make_markers(self):
        """
        Convert the dictioanry into something a that can be easy read
        :return:
        """

        # TODO need to ensure that the frame are being created correctly and fill in missing data with a flag
        to_remove = [item for item in self._data_dict.keys() if "|" in item]
        to_remove += [item for item in self._data_dict.keys() if "Trajectory Count" == item]
        for rr in to_remove:
            self._data_dict.pop(rr, None)

        for key_name, value_name in self._data_dict.iteritems():
            fixed_name = key_name[1 + key_name.find(":"):]
            self._marker_names.append(fixed_name)
            self._raw_markers[fixed_name] = []
            self._filtered_markers[fixed_name] = []

            # TODO improve this shit
            if value_name.keys()[0] == "Magnitude( X )" or value_name.keys()[0] == "Count":
                continue

            x_arr = value_name["X"]["data"]
            y_arr = value_name["Y"]["data"]
            z_arr = value_name["Z"]["data"]

            x_filt = np.convolve(x_arr, np.ones((self._filter_window,)) / self._filter_window, mode='valid')
            y_filt = np.convolve(y_arr, np.ones((self._filter_window,)) / self._filter_window, mode='valid')
            z_filt = np.convolve(z_arr, np.ones((self._filter_window,)) / self._filter_window, mode='valid')

            for inx in xrange(len(x_filt)):
                point = core.Point(x_arr[inx], y_arr[inx], z_arr[inx])
                self._raw_markers[fixed_name].append(point)
                point = core.Point(x_filt[inx], y_filt[inx], z_filt[inx])
                self._filtered_markers[fixed_name].append(point)

    def smart_sort(self, filter=False):
        """
        Gather all the frames and attempt to sort the markers into the rigid markers
        :return:
        """
        no_digits = [''.join(x for x in i if not x.isdigit()) for i in self._marker_names]  # removes digits
        single_item = list(set(no_digits))  # removes redundent items
        keys = self._marker_names

        for name in single_item:
            markers_keys = [s for s in keys if name in s]
            markers_keys.sort()
            markers = []
            for marker in markers_keys:
                if filter:
                    markers.append(self._filtered_markers[marker])
                else:
                    markers.append(self._raw_markers[marker])
            self._rigid_body[name] = markers

    def make_frame(self, _origin, _x, _y, _extra):
        """

        :param _origin:
        :param _x:
        :param _y:
        :param _extra:
        :return:
        """
        frames = []
        for o_ii, x_ii, y_ii in zip(_origin, _x, _y):
            o = np.array([o_ii.x, o_ii.y, o_ii.z]).transpose()
            x = np.array([x_ii.x, x_ii.y, x_ii.z]).transpose()
            y = np.array([y_ii.x, y_ii.y, y_ii.z]).transpose()
            xo = (x - o) / np.linalg.norm(x - o)
            yo = (y - o) / np.linalg.norm(y - o)
            zo = np.cross(xo, yo)
            xo = np.pad(xo, (0, 1), 'constant')
            yo = np.pad(yo, (0, 1), 'constant')
            zo = np.pad(zo, (0, 1), 'constant')
            p = np.pad(o, (0, 1), 'constant')
            p[-1] = 1
            F = np.column_stack((xo, yo, zo, p))
            frames.append(F)
        return frames

    def add_frame(self, name, frame):
        """

        :param name: name for the dictionary
        :param frame: frame to add the dictionary
        :return:
        """

        self._frames[name] = frame

    def auto_make_frames(self):
        """
        Auto make all the frames based on the order of the markers
        :return:
        """
        for name, value in self._rigid_body.iteritems():
            frame = self.make_frame(value[0], value[1], value[2], value[3])
            self.add_frame(name, frame)

    def auto_make_transform(self, bodies):
        """
        make the frames using the cloud method
        :param bodies:
        :return:
        """
        for name, value in self._rigid_body.iteritems():
            frames = []
            if name in bodies:

                for ii in xrange(len(value[0])):
                        frames.append(cloud_to_cloud(bodies[name], [value[0][ii], value[1][ii], value[2][ii], value[3][ii]])[0])
                self.add_frame(name, frames)

    def get_frame(self, name):
        """
        get a frame
        :param name:
        :return:
        """
        return self._frames[name]

    def get_rigid_body(self, name):
        """

        :param name:
        :return:
        """

        return self._rigid_body[name]

    def calc_joint_center(self, parent_name, child_name, start, end):
        """
        Calculate the joint center between two frames
        :param child_name:
        :param start:
        :param end:
        :return:
        """

        Tp = self.get_frame(parent_name)[start:end]
        m1 = self.get_rigid_body(child_name)[0][start:end]
        m2 = self.get_rigid_body(child_name)[1][start:end]
        m3 = self.get_rigid_body(child_name)[2][start:end]
        m4 = self.get_rigid_body(child_name)[3][start:end]
        m = [m1, m2, m3, m4]

        global_joint = calc_CoR(m)

        axis = calc_AoR(m)
        local_joint = np.array([[0.0], [0.0], [0.0], [0.0]])

        for T in Tp:
            local_joint += transform_vector(np.linalg.pinv(T), global_joint )/len(Tp)

        return np.vstack((global_joint, [1])), axis, local_joint

    def play(self, joints=None, save=False, name="im"):
        """
        play the markers
        :param joints:
        :param save:
        :return:
        """

        x_total = []
        y_total = []
        z_total = []
        joints_points = []
        fps = 100  # Frame per sec
        keys = self._filtered_markers.keys()
        nfr = len(self._filtered_markers[keys[0]])  # Number of frames

        for frame in xrange(nfr):
            x = []
            y = []
            z = []
            for key in keys:
                point = self._filtered_markers[key][frame]
                x += [point.x]
                y += [point.y]
                z += [point.z]
            x_total.append(x)
            y_total.append(y)
            z_total.append(z)
            x = []
            y = []
            z = []
            if joints is not None:
                for joint in joints:
                    x.append(joint[frame][0])
                    y.append(joint[frame][1])
                    z.append(joint[frame][2])
                joints_points.append([x, y, z])

        self._fig = plt.figure()
        self._ax = self._fig.add_subplot(111, projection='3d')
        self._ax.set_autoscale_on(False)

        ani = animation.FuncAnimation(self._fig,
                                      self.__animate, nfr,
                                      fargs=(x_total, y_total, z_total, joints_points),
                                      interval=100 / fps)
        if save:
            Writer = animation.writers['ffmpeg']
            writer = Writer(fps=15, metadata=dict(artist='Me'), bitrate=1800)
            ani.save(name + '.mp4', writer=writer)
            plt.show()
        else:
            plt.show()

    def __animate(self, frame, x, y, z, centers=None):
        """

        :param frame:
        :param x:
        :param y:
        :param z:
        :param centers:
        :return:
        """

        self._ax.clear()
        self._ax.set_xlabel('X Label')
        self._ax.set_ylabel('Y Label')
        self._ax.set_zlabel('Z Label')
        self._ax.axis([-500, 500, -200, 3000])
        self._ax.set_zlim3d(0, 1250)
        self._ax.scatter(x[frame], y[frame], z[frame], c='r', marker='o')
        if len(centers) > 0:
            self._ax.scatter(centers[frame][0], centers[frame][1], centers[frame][2], c='g', marker='o')


def transform_markers(transforms, markers):
    """

    :param transforms:
    :param markers:
    :return:
    """
    trans_markers = []
    for marker in markers:
        adjusted_locations = []
        for transform, frame in zip(transforms, marker):
            v = np.array([[frame.x, frame.y, frame.z, 1.0]]).T
            v_prime = np.dot(transform, v)
            new_marker = core.Point(v_prime[0][0], v_prime[1][0], v_prime[2][0])
            adjusted_locations.append(new_marker)
        trans_markers.append(adjusted_locations)
    return trans_markers


def make_frame(markers):
    """
    Create a frame based on the marker layout
    :param markers:
    :return:
    """
    origin = markers[0]
    x_axis = markers[1]
    y_axis = markers[2]

    xo = origin - x_axis
    yo = origin - y_axis
    zo = np.cross(xo, yo)
    xo = np.pad(xo, (0, 1), 'constant')
    yo = np.pad(yo, (0, 1), 'constant')
    zo = np.pad(zo, (0, 1), 'constant')
    p = np.pad(origin, (0, 1), 'constant')
    p[-1] = 1
    return np.column_stack((xo, yo, zo, p))


def get_all_transformation_to_base(parent_frames, child_frames):
    """

    :type world_to_base_frame: np.array
    :param world_to_base_frame:
    :param body_frames:
    :return:
    """

    frames = []
    for parent, child in zip(parent_frames, child_frames):
        frames.append(get_transform_btw_two_frames(parent, child))

    return frames


def get_transform_btw_two_frames(parent_frame, child_frame):
    """

    :param parent_frame:
    :param child_frame:
    :return:
    """
    return np.linalg.inv(parent_frame) * child_frame


def get_angle_between_vects(v1, v2):
    """
    returns the angle between two vectors
    https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249
    :param v1: vector 1
    :param v2: vector 2
    :return:
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def transform_vector(frame, vector):
    """
    transform a vector from one frame to another
    :param frame:
    :param vector:
    :return:
    """
    p = np.vstack((vector, [1]))
    return np.dot(frame, p)


def batch_transform_vector(frames, vector):
    """

    :param frames: list of frames
    :param vector: vector to transform
    :return:
    """

    trans_vectors = []

    for T in frames:
        p = np.dot(T, vector)
        #p = np.dot(np.eye(4), vector)
        trans_vectors.append(p)

    return trans_vectors

def unit_vector(vector):
    """
    Returns the unit vector of the vector.
    :param vector:
    :return:
    """
    return vector / np.linalg.norm(vector)


def avg_vector(markers):
    """
    averages the marker location based
    :param markers: a marker
    :return: norm of all the markers
    """
    vp_norm = []
    for marker in markers:
        vp = np.array((0.0, 0.0, 0.0))
        for point in marker:
            vp = vp + np.array((point.x, point.y, point.z))
        vp /= len(marker)
        vp_norm.append(vp)
    return vp_norm


def calc_CoR(markers):
    '''
        Calculate the center of rotation given two data
        sets representing two frames on separate rigid bodies connected by a
        spherical joint. The function calculates the position of the CoR in the
        reference rigidi body frame
        For more information on this derivation see "New Least Squares Solutions
        for Estimating the Average Centre of Rotation and the Axis of Rotation"
        by Sahan S. Hiniduma
    '''

    A = calc_A(markers)
    b = calc_b(markers)
    Ainv = np.linalg.pinv(2.0 * A)
    return np.dot(Ainv, b)


def calc_AoR(markers):
    """
        Calculate the center of rotation given two data
        sets representing two frames on separate rigid bodies connected by a
        spherical joint. The function calculates the position of the CoR in the
        reference rigidi body frame
        For more information on this derivation see "New Least Squares Solutions
        for Estimating the Average Centre of Rotation and the Axis of Rotation"
        by Sahan S. Hiniduma


    :type markers: list
    :param markers: list of markers, each marker is a list of core.Exoskeleton.Points
    :return: axis of rotation
    :rtype np.array
    """
    A = calc_A(markers)  # calculates the A matrix
    E_vals, E_vecs = np.linalg.eig(
        A)  # I believe that the np function eig has a different output than the matlab function eigs
    min_E_val_idx = np.argmin(E_vals)
    axis = E_vecs[:, min_E_val_idx]
    return axis


def calc_A(markers):
    """

    :param markers: array of markers
    :return: A array
    """

    A = np.zeros((3, 3))
    vp_norm = avg_vector(markers)
    for marker, vp_n in zip(markers, vp_norm):  # loop though each marker
        Ak = np.zeros((3, 3))
        for point in marker:  # go through is location of the marker
            v = np.array((point.x, point.y, point.z))
            Ak = Ak + v.reshape((-1, 1)) * v
        Ak = (1.0 / len(marker)) * Ak - vp_n.reshape((-1, 1)) * vp_n
        A = A + Ak
    return A


def calc_b(markers):
    """

    :param markers: array of markers
    :return: b array
    """
    b = np.array((0.0, 0.0, 0.0))
    vp_norm = avg_vector(markers)
    for ii, marker in enumerate(markers):
        invN = 1.0 / len(marker)
        v2_sum = 0
        v3_sum = np.array((0.0, 0.0, 0.0))
        for point in marker:
            # v = np.array((point.x, point.y, point.z))
            # print np.dot(v.reshape((-1,1)),v.reshape((-1,1)))
            v2 = (point.x * point.x + point.y * point.y + point.z * point.z)
            v2_sum = v2_sum + invN * v2
            v3_sum = v3_sum + invN * (v2 * np.array((point.x, point.y, point.z)))
        b = b + v3_sum - v2_sum * vp_norm[ii]

    return b.reshape((-1, 1))


def cloud_to_cloud(A_, B_):
    """
    Get the transformation between two frames of marker sets.
    http://nghiaho.com/?page_id=671
    :param A_: ,rigid body markers set
    :param B_: currnet position of the markers
    :return:
    """
    A = np.asmatrix(points_to_matrix(A_))
    B = np.asmatrix(points_to_matrix(B_))

    assert len(A) == len(B)

    N = A.shape[0];  # total points

    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # centre the points
    AA = A - np.tile(centroid_A, (N, 1))
    BB = B - np.tile(centroid_B, (N, 1))

    # dot is matrix multiplication for array
    H = np.transpose(AA) * BB

    U, S, Vt = np.linalg.svd(H)

    R = Vt.T * U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T * U.T

    p = -R * centroid_A.T + centroid_B.T

    A2 = (R * A.T) + np.tile(p, (1, N))
    A2 = A2.T
    err = A2 - B
    err = np.multiply(err, err)
    err = sum(err)
    rmse = sqrt(err / N)

    T = np.zeros((4, 4))
    T[:3, :3] = R
    for ii in xrange(3):
        T[ii, 3] = p[ii]
    T[3, 3] = 1.0


    return T, rmse


def get_center(markers, R):
    """
    Get the marker set
    :param markers:
    :param R:
    :return:
    """

    x1 = np.array((markers[0][0].x, markers[0][0].y, markers[0][0].z)).reshape((-1, 1))
    x2 = np.array((markers[1][0].x, markers[1][0].y, markers[1][0].z)).reshape((-1, 1))
    xc = -np.dot(np.linalg.pinv(R + np.eye(3)), (x2 - np.dot(R, x1)))

    return xc


def minimize_center(vectors, axis, initial):
    """
    Optimize the center of the rotation of the axis by finding the closest point
    in the line to the frames
    :param vectors:
    :param axis:
    :param initial:
    :return:
    """

    def objective(x):
        C = 0
        for vect in vectors:
            C += np.sqrt(np.power((x[0] - vect[0]), 2) + np.power((x[1] - vect[1]), 2) + np.power((x[2] - vect[2]), 2))
        return C

    def constraint(x):
        return np.array((x[0], x[1], x[2])) - x[3] * axis - initial

    N = 1000
    b = (-N, N)
    bnds = (b, b, b, b)
    con = {'type': 'eq', 'fun': constraint}
    cons = ([con])
    solution = minimize(objective, np.append(initial, 0), method='SLSQP', \
                        bounds=bnds, constraints=cons)
    return solution


def calc_mass_vect(markers):
    """
    find the average vector to  frame
    :param markers:
    :return:
    """
    x = 0
    y = 0
    z = 0
    for point in markers:
        x += point.x
        y += point.y
        z += point.z

    vect = np.array((x / len(markers),
                     y / len(markers),
                     z / len(markers)))
    return vect


def calc_vector_between_points(start_point, end_point):
    """
    calculate the vector between two points
    :param start_point: first point
    :param end_point: sencound point
    :return:
    """
    return end_point - start_point


def get_distance(point1, point2):
    """
    Get the distance between two points
    :param point1: first point
    :param point2: secound point
    :return:
    """
    return np.sum(np.sqrt(np.power(calc_vector_between_points(point1, point2), 2)))


def R_to_axis_angle(matrix):
    """Convert the rotation matrix into the axis-angle notation.
    Conversion equations
    ====================
    From Wikipedia (http://en.wikipedia.org/wiki/Rotation_matrix), the conversion is given by::
        x = Qzy-Qyz
        y = Qxz-Qzx
        z = Qyx-Qxy
        r = hypot(x,hypot(y,z))
        t = Qxx+Qyy+Qzz
        theta = atan2(r,t-1)
    @param matrix:  The 3x3 rotation matrix to update.
    @type matrix:   3x3 numpy array
    @return:    The 3D rotation axis and angle.
    @rtype:     numpy 3D rank-1 array, float
    """

    # Axes.
    axis = np.zeros(3)
    axis[0] = matrix[2, 1] - matrix[1, 2]
    axis[1] = matrix[0, 2] - matrix[2, 0]
    axis[2] = matrix[1, 0] - matrix[0, 1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    theta = math.atan2(r, t - 1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta


def sphereFit(frames):
    """
    Fit a sphere to a serise of transformations
    :param frames:
    :return:
    """
    spX = []
    spY = []
    spZ = []
    for frame in frames:
        spX.append(frame[0])
        spY.append(frame[1])
        spZ.append(frame[2])

    spX = np.array(spX)
    spY = np.array(spY)
    spZ = np.array(spZ)
    A = np.zeros((len(spX), 4))
    A[:, 0] = spX * 2
    A[:, 1] = spY * 2
    A[:, 2] = spZ * 2
    A[:, 3] = 1

    #   Assemble the f matrix
    f = np.zeros((len(spX), 1))
    f[:, 0] = (spX * spX) + (spY * spY) + (spZ * spZ)
    C, residules, rank, singval = np.linalg.lstsq(A, f)

    #   solve for the radius
    t = (C[0] * C[0]) + (C[1] * C[1]) + (C[2] * C[2]) + C[3]
    radius = np.sqrt(t)
    return radius, C[:3]


def points_to_matrix(points):
    """
    converts the points to an array
    :param points:
    :return:
    """

    cloud = np.zeros((len(points), 3))
    for index, point in enumerate(points):
        cloud[index, :] = [point.x, point.y, point.z]

    return cloud


def get_rmse(marker_set, body):
    """
    Get the RMSE of the transform and a body location
    :param marker_set:
    :param body:
    :return:
    """
    error = []
    for frame in xrange(1000):
        f = [body[0][frame], body[1][frame], body[2][frame], body[frame]]
        T, err = Markers.cloud_to_cloud(marker_set, f)
        error.append(err)


def fit_to_plane(points):
    """
    fit a plan to an array of points using regression
    https://math.stackexchange.com/questions/99299/best-fitting-plane-given-a-set-of-points
    :param points: list of points
    :return:
    """

    tmp_A = []
    tmp_b = []
    for point in points:
        tmp_A.append([point.x, point.y, 1])
        tmp_b.append(point.z)
    b = np.matrix(tmp_b).T
    A = np.matrix(tmp_A)
    fit = (A.T * A).I * A.T * b

    errors = b - A * fit
    residual = np.linalg.norm(errors)
    fit = unit_vector(fit)
    return fit, residual


if __name__ == '__main__':
    DataSets1 = [core.Point(531.6667, - 508.9951, 314.4273),
                 core.Point(510.5082, - 457.7791, 357.1969),
                 core.Point(463.9945, - 476.0904, 356.1137),
                 core.Point(552.4579, - 566.4891, 393.5611),
                 core.Point(505.9442, - 584.8004, 392.4779)]

    DataSets2 = [[-55.4398, 406.9759, - 487.4170],
                 [-117.4716, 384.3339, -510.7755],
                 [-99.5008, 336.9028, - 511.4401],
                 [-84.8805, 394.2636, - 393.6067],
                 [-67.3354, 347.3059, - 393.7805]]

    marker = [core.Point(0.0, 50.0, 0.0),
              core.Point(-70.0, 50.0, 0.0),
              core.Point(-70, 0, 0),
              core.Point(0.0, 50.0, 100.0),
              core.Point(0.0, 0.0, 100.0)]

    # print cloudtocloud(marker, DataSets1)

    # marker0 = np.asarray([3.6, 5.4, 1.69]).transpose()
    # marker1 = np.asarray([4.0, 6.0, 1.75]).transpose()
    # marker2 = np.asarray([3.8, 7.2, 1.59]).transpose()
    # marker3 = np.asarray([3.4, 7.9, 1.34]).transpose()
    #
    # frame = np.asarray([marker0, marker1, marker2, marker3])
    # make_frame(frame)
    # vect = get_angle_between_vects(marker1, marker2)
    # print transform_vector(frame, marker0)
