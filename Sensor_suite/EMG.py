from Sensor_suite import PostSensor


class EMG(PostSensor.PostSensor):

    def __init__(self, name, sensor):
        super(EMG, self).__init__(name, sensor, "EMG")

    def get(self):
        return self.sensor["IM EMG"]
