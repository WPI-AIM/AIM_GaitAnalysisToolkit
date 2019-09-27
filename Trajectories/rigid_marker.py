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

    vp_norm = []
    for marker in markers:
        vp = np.array((0.0, 0.0, 0.0))
        for point in marker:
            vp = vp + np.array((point.x, point.y, point.z))
        vp /= len(marker)
        vp_norm.append(vp)
    return vp_norm

def find_CoR(frame):
    '''
        Calculate the center of rotation given two data
        sets representing two frames on separate rigid bodies connected by a
        spherical joint. The function calculates the position of the CoR in the
        reference rigidi body frame
        For more information on this derivation see "New Least Squares Solutions
        for Estimating the Average Centre of Rotation and the Axis of Rotation"
        by Sahan S. Hiniduma
    '''

    A = get_A(frame)
    b = get_b(frame)
    print A
    print b
    return np.linalg.solve(A,b)

def get_A(frame):
    """

    :param frame: array of markers
    :return: A array
    """

    A = np.zeros((3, 3))
    vp_norm = avg_vector(frame)
    for marker, vp_n in zip(frame, vp_norm): # loop though each marker
        Ak = np.zeros((3, 3))
        for point in marker: # go through is location of the marker
            v = np.array((point.x, point.y, point.z))
            Ak = Ak + v.reshape((-1, 1)) * v
        Ak = (1.0 / len(marker)) * Ak - vp_n.reshape((-1, 1)) * vp_n
        A = A + Ak
    return 2.0*A

def get_b(frame):
    """

    :param frame: array of markers
    :return: b array
    """
    b = np.array((0.0, 0.0, 0.0))
    vp_norm = avg_vector(frame)

    for ii, marker in enumerate(frame):
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

if __name__ == '__main__':

    marker0 = np.asarray([3.6, 5.4, 1.69]).transpose()
    marker1 = np.asarray([4.0 , 6.0, 1.75 ]).transpose()
    marker2 = np.asarray([3.8, 7.2, 1.59]).transpose()
    marker3 = np.asarray([3.4, 7.9, 1.34]).transpose()

    frame = np.asarray([marker0, marker1, marker2, marker3])
    make_frame(frame)
    vect = get_angle_between_vects(marker1, marker2)
    print vect
    print transform_vector(frame, marker0)

def find_AoR(frame):
    A = get_A(frame)
    b = get_b(frame)
    w, v = np.linalg.eig(A)
    print "w ",  w
    print "v ",  v



