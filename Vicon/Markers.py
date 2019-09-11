from lib.Exoskeleton.Robot import core
import numpy as np

class Markers(object):

    def __init__(self, marker_dict):
        self._data_dict = marker_dict
        self._raw_markers = {}
        self._rigid_body = {}
        self._marker_names = []

    def make_markers(self):

        for key_name, value_name in self._data_dict.iteritems():
            self._marker_names.append(key_name)
            x = value_name["X"]["data"]
            y = value_name["Y"]["data"]
            z = value_name["z"]["data"]
            point = core.Point(x, y, z)
            self._raw_markers[key_name] = point

    def smart_sort(self):

        no_digits = [''.join(x for x in i if x.isalpha()) for i in self._data_dict.keys()] # removes digits
        single_item = list(set(no_digits)) # removes redundent items
        for name in single_item:
           self._rigid_body[name] = [s for s in self._data_dict.keys if name in s]

    def make_frame(self,name, origin, x, y, extra):

        for o, x, y in zip(origin,x, y):
            o = np.array([o.x, o.y, o.z ]).transpose()
            x = np.array([x.x, x.y, x.z]).transpose()
            y = np.array([x.x, y.y, y.z]).transpose()
            xo = x - origin
            yo = y - origin
            zo = np.cross(xo, yo)





