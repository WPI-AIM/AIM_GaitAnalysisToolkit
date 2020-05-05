import numpy
import abc


_body_lenths = {}
_body_masses = {}

_body_lenths["foot_length"] = 0.152
_body_lenths["foot_breadth"] = 0.152
_body_lenths["ankle"] = 0.039
_body_lenths["shank"] = 0.2811
_body_lenths["thigh"] = 0.245



class Model(object):

    def __init__(self, mass, height):

        self._mass = mass
        self._height = height
        self.lengths = {}
        self.mass = {}

    def make_body(self):

        keys = _body_lenths.keys()

        for key in keys:
            self.lengths[key] = _body_lenths[key]*self._height


