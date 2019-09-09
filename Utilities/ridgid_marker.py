import numpy as np



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
    print F


def get_all_transformation_to_base(parent_frame, child_frames):
    """

    :type world_to_base_frame: np.array
    :param world_to_base_frame:
    :param body_frames:
    :return:
    """
    frames = []

    for frame in child_frames:
        frames.append(get_transform(parent_frame, frame))

    return frames

def get_transform(parent_frame, child_frame):
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


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)


# def get_A(markers):
#
#     A = 0
#     for marker in markers:
#         inner_sum = 0
#         N = len(marker)
#         vbar = 0.0
#         for frame in marker:
#             inner_sum += (1./N) * (frame * np.transpose(frame))
#             vbar = frame*(1./N)
#         A += inner_sum - vbar*np.transpose(vbar)
#     return 2*A
#
#
# def get_B(markers):
#     B = None
#     vbar = np.asarray([ [0.0], [0.0], [0.0]  ])
#     for marker in markers:
#         inner_sum = 0
#         N = len(marker)
#         vbar = 0.0
#         for frame in marker:
#             inner_sum += (1. / N) * (frame * np.transpose(frame))
#             vbar = frame * (1. / N)
#         Ba += inner_sum - vbar * np.transpose(vbar)
#     return 2 * A



if __name__ == '__main__':

    marker0 = np.asarray([3.6, 5.4, 1.69]).transpose()
    marker1 = np.asarray([4.0 , 6.0, 1.75 ]).transpose()
    marker2 = np.asarray([3.8, 7.2, 1.59]).transpose()
    marker3 = np.asarray([3.4, 7.9, 1.34]).transpose()

    frame = np.asarray( [ marker0, marker1, marker2, marker3])
    make_frame(frame)





