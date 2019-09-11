from lib.Exoskeleton.Robot import core
import numpy as np

class Markers(object):

    def __init__(self, marker_dict):
        self._data_dict = marker_dict
        self._raw_markers = {}
        self._rigid_body = {}
        self._marker_names = []
        self._frames = {}

    @property
    def marker_names(self):
        return self._marker_names

    def make_markers(self):

        for key_name, value_name in self._data_dict.iteritems():
            self._marker_names.append(key_name)
            x = value_name["X"]["data"]
            y = value_name["Y"]["data"]
            z = value_name["Z"]["data"]
            point = core.Point(x, y, z)
            self._raw_markers[key_name] = point

    def smart_sort(self):

        no_digits = [''.join(x for x in i if x.isalpha()) for i in self._data_dict.keys()] # removes digits
        single_item = list(set(no_digits)) # removes redundent items
        for name in single_item:
           self._rigid_body[name] = [s for s in self._data_dict.keys if name in s]

    def make_frame(self, origin, x, y, extra):
        Frames = []
        for o, x, y in zip(origin,x, y):
            o = np.array([o.x, o.y, o.z ]).transpose()
            x = np.array([x.x, x.y, x.z]).transpose()
            y = np.array([x.x, y.y, y.z]).transpose()
            xo = x - origin
            yo = y - origin
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
        self._frames[name] = frame

    def auto_make_frames(self):
        print "sad;ljas;ldfj "
        for name, value in self._rigid_body.iteritems():
            print "here"
            frame = self.make_frame(value[0], value[1], value[2], value[3])
            print "name ", name
            self.add_frame(name, frame)

    def get_frame(self, name):
        return self._frames[name]



