from lib.Exoskeleton.Robot import core
import numpy as np

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
            frame = self.make_frame(value[0], value[1], value[2], value[3])
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



