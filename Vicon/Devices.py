class Devices(object):

    def __init__(self, name, sensor, type):
        self._name = name
        self._sensor = sensor
        self.type = type
        self.offset = 20

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        self._name = value

    @property
    def sensor(self):
        return self._sensor

    @sensor.setter
    def sensor(self, value):
        self._sensor = value

    @property
    def type(self):
        return self._name

    @type.setter
    def type(self, value):
        self._type = value

    def get_offset_index(self, dx):
        return dx * self.offset

    def get(self):
        pass
