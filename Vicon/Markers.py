from lib.Exoskeleton.Robot import core
import numpy as np
from scipy.optimize import minimize
import math
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
    @property
    def marker_names(self):
        """
        :return marker names
        :return:
        """
        return self._marker_names

    def make_markers(self):
        """
        Convert the dictioanry into something a that can be easy read
        :return:
        """
        for key_name, value_name in self._data_dict.iteritems():
            self._marker_names.append(key_name)
            self._raw_markers[key_name] = []
            x_arr = value_name["X"]["data"]
            y_arr = value_name["Y"]["data"]
            z_arr = value_name["Z"]["data"]
            for inx in xrange(len(value_name["X"]["data"])):
                x = x_arr[inx]
                y = y_arr[inx]
                z = z_arr[inx]
                point = core.Point(x, y, z)
                self._raw_markers[key_name].append(point)

    def smart_sort(self):
        """
        Gather all the frames and attempt to sort the markers into the rigid markers
        :return:
        """
        no_digits = [''.join(x for x in i if not x.isdigit() ) for i in self._data_dict.keys()] # removes digits
        single_item = list(set(no_digits)) # removes redundent items
        keys = self._data_dict.keys()
        for name in single_item:
            markers_keys = [s for s in keys if name in s]
            markers = []
            for marker in markers_keys:
                markers.append(self._raw_markers[marker])
            self._rigid_body[name] = markers


    def make_frame(self, _origin, _x, _y, _extra):
        Frames = []
        for o_ii, x_ii, y_ii in zip(_origin, _x, _y):
            o = np.array([o_ii.x, o_ii.y, o_ii.z ]).transpose()
            x = np.array([x_ii.x, x_ii.y, x_ii.z]).transpose()
            y = np.array([y_ii.x, y_ii.y, y_ii.z]).transpose()
            xo = (x - o) / np.linalg.norm(x - o)
            yo = (y - o) / np.linalg.norm(y - o)
            zo = np.cross(xo, yo)
            xo = np.pad(xo, (0,1), 'constant')
            yo = np.pad(yo, (0, 1), 'constant')
            zo = np.pad(zo, (0, 1), 'constant')
            p = np.pad(o, (0,1), 'constant')
            p[-1] = 1
            F = np.column_stack((xo, yo, zo, p))
            Frames.append(F)
        return Frames

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
            frame = self.make_frame(value[0], value[3], value[2], value[1])
            self.add_frame(name, frame)

    def get_frame(self, name):
        """
        get a frame
        :param name:
        :return:
        """
        return self._frames[name]

    def get_rigid_body(self, name):
        return self._rigid_body[name]




import numpy as np
from lib.Exoskeleton.Robot import core

def transform_markers(transforms, markers):
    trans_markers = []
    for marker in markers:
        adjusted_locations = []
        for transform, frame in zip(transforms, marker):
            v = np.array([[frame.x, frame.y,frame.z,1.0]]).T
            v_prime = np.dot( np.linalg.inv(transform),v)
            new_marker = core.Point(v_prime[0][0],v_prime[1][0],v_prime[2][0])
            adjusted_locations.append(new_marker)
        trans_markers.append(adjusted_locations)
    return trans_markers

def make_frame(markers):

    origin = markers[0]
    x_axis = markers[1]
    y_axis = markers[2]

    xo = origin - x_axis
    yo = origin - y_axis
    zo = np.cross(xo,yo)
    xo = np.pad(xo, (0, 1), 'constant')
    yo = np.pad(yo, (0, 1), 'constant')
    zo = np.pad(zo, (0, 1), 'constant')
    p = np.pad(origin, (0, 1), 'constant')
    p[-1] = 1
    F = np.column_stack((xo,yo,zo,p))

def get_all_transformation_to_base(parent_frames, child_frames):
    """

    :type world_to_base_frame: np.array
    :param world_to_base_frame:
    :param body_frames:
    :return:
    """

    frames = []
    count = 0
    for parent, child in zip(parent_frames, child_frames):
        frames.append(get_transform_btw_two_frames(parent, child))

    return frames

def get_transform_btw_two_frames(parent_frame, child_frame):
    return np.linalg.inv(parent_frame)*child_frame

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
    p = np.pad(vector, (0, 1), 'constant')
    p[-1] = 1
    return frame*p

def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
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
    Ainv = np.linalg.pinv(2.0*A)
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
    A = calc_A(markers) #calculates the A matrix
    E_vals, E_vecs = np.linalg.eig(A) # I believe that the np function eig has a different output than the matlab function eigs
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
    for marker, vp_n in zip(markers, vp_norm): # loop though each marker
        Ak = np.zeros((3, 3))
        for point in marker: # go through is location of the marker
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
    print vp_norm
    for ii, marker in enumerate(markers):
        invN = 1.0/len(marker)
        v2_sum = 0
        v3_sum = np.array((0.0, 0.0, 0.0))
        for point in marker:
            #v = np.array((point.x, point.y, point.z))
            #print np.dot(v.reshape((-1,1)),v.reshape((-1,1)))
            v2 = (point.x*point.x + point.y*point.y + point.z*point.z)
            v2_sum = v2_sum + invN * v2
            v3_sum = v3_sum + invN * (v2 * np.array((point.x, point.y, point.z)))
        b = b + v3_sum - v2_sum*vp_norm[ii]

    return b.reshape((-1, 1))

def get_transformation(markers):
    """
    Get the transforms between marker at different times
    :param markers:
    :return:
    """
    A = np.matrix((markers[0][0].x, markers[0][0].y, markers[0][0].z))
    B = np.matrix((markers[1][0].x, markers[1][0].y, markers[1][0].z))

    N = A.shape[0]
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
        print "Reflection detected"
        Vt[2, :] *= -1
        R = Vt.T * U.T

    t = -R * centroid_A.T + centroid_B.T

    A2 = R*A.T + t

    err = A2 - B.T

    err = np.multiply(err, err)
    err = sum(err)
    rmse = np.sqrt(err / N)

    return R, t

def get_center(markers, R):
    print markers[0]
    x1 = np.array((markers[0][0].x, markers[0][0].y, markers[0][0].z)).reshape((-1,1))
    print x1
    x2 = np.array((markers[1][0].x, markers[1][0].y, markers[1][0].z)).reshape((-1,1))
    xc = -np.dot(np.linalg.pinv(R + np.eye(3)), (x2 - np.dot(R, x1)))

    return xc

def minimize_center(vectors, axis, initial ):
    # optimize

    def objective(x):
        C = 0
        for vect in vectors:
            C += np.sqrt( np.power((x[0] - vect[0]),2) + np.power((x[1] - vect[1]),2) + np.power((x[2] - vect[2]),2) )
        return C

    def constraint(x):
        return np.array((x[0], x[1], x[2])) - x[3]*axis - initial

    N = 1000
    b = (-N, N)
    bnds = (b, b, b, b)
    con= {'type': 'eq', 'fun': constraint}
    cons = ([con])
    solution = minimize(objective, np.append(initial,0), method='SLSQP', \
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

    vect = np.array((x/len(markers), y/len(markers), z/len(markers)))
    return vect

def calc_vector(start_point, end_point):
    """
    calculate the vector between two points
    :param start_point:
    :param end_point:
    :return:
    """
    return end_point - start_point

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
    axis[0] = matrix[2,1] - matrix[1,2]
    axis[1] = matrix[0,2] - matrix[2,0]
    axis[2] = matrix[1,0] - matrix[0,1]

    # Angle.
    r = np.hypot(axis[0], np.hypot(axis[1], axis[2]))
    t = matrix[0, 0] + matrix[1, 1] + matrix[2, 2]
    theta = math.atan2(r, t-1)

    # Normalise the axis.
    axis = axis / r

    # Return the data.
    return axis, theta

if __name__ == '__main__':

    marker0 = np.asarray([3.6, 5.4, 1.69]).transpose()
    marker1 = np.asarray([4.0 , 6.0, 1.75 ]).transpose()
    marker2 = np.asarray([3.8, 7.2, 1.59]).transpose()
    marker3 = np.asarray([3.4, 7.9, 1.34]).transpose()

    frame = np.asarray([marker0, marker1, marker2, marker3])
    make_frame(frame)
    vect = get_angle_between_vects(marker1, marker2)
    print transform_vector(frame, marker0)


