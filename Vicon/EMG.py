import Devices


class EMG(Devices.Devices):

    def __init__(self, name, sensor):
        super(EMG, self).__init__(name, sensor, "EMG")

    def get(self):
        return self.sensor["data"]

    def get_unit(self):
        return self.sensor["unit"]
