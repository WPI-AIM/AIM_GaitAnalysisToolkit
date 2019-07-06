from Utilities import utilities


class Vicon(object):

    def __init__(self, file_path):
        self._file_path = file_path
        self.output_names = ["Devices", "Joints", "Model Outputs", "Segments", "Trajectories"]
        self.data_dict = utilities.open_vicon_file(self._file_path, self.output_names)

    def _filter_dict(self, sensors, substring):
        return list(filter(lambda x: substring in x, sensors.keys()))

    def get_force_plates(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'Force') + ['Combined Moment'] + ['Combined CoP']
        return dict([(key, sensors[key]) for key in keys])

    def get_EMGs(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'EMG')
        return dict([(key, sensors[key]) for key in keys])

    def get_EMGs(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'EMG')
        return dict([(key, sensors[key]) for key in keys])

    def get_IMUs(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'IMU')
        return dict([(key, sensors[key]) for key in keys])

    def get_IMUs(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'IMU')
        return dict([(key, sensors[key]) for key in keys])

    def get_Accelerometers(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'Accelerometers')
        return dict([(key, sensors[key]) for key in keys])

    def fix_keys(self):
        self.data_dict
