import Accel
import EMG
import ForcePlate
import IMU
from Utilities import utilities


class Vicon(object):

    def __init__(self, file_path):
        self._file_path = file_path
        self.output_names = ["Devices", "Joints", "Model Outputs", "Segments", "Trajectories"]
        self.data_dict = utilities.open_vicon_file(self._file_path, self.output_names)
        self._T_EMGs = {}
        self._EMGs = {}
        self._force_plates = {}
        self._IMUs = {}
        self._accel = {}

    def _filter_dict(self, sensors, substring):
        return list(filter(lambda x: substring in x, sensors.keys()))

    def _make_force_plates(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'Force_Plate')  # + ['Combined Moment'] + ['Combined CoP']
        self._force_plates[1] = ForcePlate.ForcePlate("Force_Plate_1", sensors["Force_Plate__Force_1"],
                                                      sensors["Force_Plate__Moment_1"])

        self._force_plates[2] = ForcePlate.ForcePlate("Force_Plate_2", sensors["Force_Plate__Force_2"],
                                                      sensors["Force_Plate__Moment_2"])

    def _make_EMGs(self):
        sensors = self.data_dict["Devices"]
        all_keys = self._filter_dict(sensors, 'EMG')
        T_EMG_keys = self._filter_dict(sensors, 'T_EMG')
        EMG_keys = [x for x in all_keys if x not in T_EMG_keys]
        for e_key, t_key in zip(EMG_keys, T_EMG_keys):
            self._T_EMGs[int(filter(str.isdigit, t_key))] = EMG.EMG(t_key, sensors[t_key])
            self._EMGs[int(filter(str.isdigit, e_key))] = EMG.EMG(e_key, sensors[e_key])
        # return dict([(key, sensors[key]) for key in keys])

    def _make_IMUs(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'IMU')
        for key in keys:
            self._IMUs[int(filter(str.isdigit, key))] = IMU.IMU(key, sensors[key])

    def _make_Accelerometers(self):
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'Accelerometers')
        for key in keys:
            self._accels[int(filter(str.isdigit, key))] = Accel.Accel(key, sensors[key])

    @property
    def accel(self):
        return self._accel

    @property
    def force_plate(self):
        return self._force_plates

    @property
    def IMUs(self):
        return self._IMUs

    @property
    def T_EMGs(self):
        return self._T_EMGs

    @property
    def EMGs(self):
        return self._EMGs

    def get_imu(self, index):
        return self.IMUs[index]

    def get_accel(self, index):
        return self.accels[index]

    def get_force_plate(self, index):
        return self.force_plate[index]

    def get_emg(self, index):
        return self.EMGs[index]

    def get_t_emg(self, index):
        return self.T_EMGs[index]


if __name__ == '__main__':
    file = "/home/nathaniel/git/Gait_Analysis_Toolkit/Utilities/Walking01.csv"
    data = Vicon(file)
    print data.get_EMGs().keys()
