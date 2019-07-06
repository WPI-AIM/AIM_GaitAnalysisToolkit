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

    @property
    def accel(self):
        """
        Get the Accels dict
        :return: Accels
        :type: dict
        """
        return self._accel

    @property
    def force_plate(self):
        """
         Get the force plate dict
         :return: Force plates
         :type: dict
        """
        return self._force_plates

    @property
    def IMUs(self):
        """
         Get the IMU dict
         :return: IMU
         :type: dict
        """
        return self._IMUs

    @property
    def T_EMGs(self):
        """
         Get the EMG dict
         :return: T EMG
         :type: dict
        """
        return self._T_EMGs

    @property
    def EMGs(self):
        """
        Get the EMGs dict
        :return: EMGs
        :type: dict
        """
        return self._EMGs

    def get_model_output(self):
        """
        get the model output
        :return: model outputs
        :rtype: dict
        """
        return self.data_dict["Model Outputs"]

    def get_segments(self):
        """
        get the segments
        :return: model segments
        :type: dict
        """
        return self.data_dict["Segments"]

    def get_markers(self):
        """
        get the markers
        :return: markers
        :type: dict
        """
        return self.data_dict["Trajectories"]

    def get_joints(self):
        """
        get the joints
        :return: model joints
        :type: dict
        """
        return self.data_dict["Joints"]

    def get_imu(self, index):
        """
        get the a imu
        :param index: imu number
        :return: imu
        :type: IMU.IMU
        """
        return self.IMUs[index]

    def get_accel(self, index):
        """
        get the a Accel
        :param index: Accel number
        :return: Accel
        :type: Accel.Accel
        """
        return self.accels[index]

    def get_force_plate(self, index):
        """
        get the a force plate
        :param index: force plate number
        :return: Force plate
        :type: ForcePlate.ForcePlate
        """
        return self.force_plate[index]

    def get_emg(self, index):
        """
       Get the EMG values
       :param index: number of sensor
       :return: EMG
       :rtype: EMG.EMG
        """
        return self.EMGs[index]

    def get_t_emg(self, index):
        """
        Get the T EMG values
        :param index: number of sensor
        :return: EMG
        :rtype: EMG.EMG
        """
        return self.T_EMGs[index]

    def _filter_dict(self, sensors, substring):
        """
        filter the dictionary
        :param sensors: Dictionary to parse
        :param substring: substring of the keys to look for in the dict
        :return: keys that contain the substring
        :type: list
        """
        return list(filter(lambda x: substring in x, sensors.keys()))

    def _make_force_plates(self):
        """
        generate force plate models
        :return: None
        """
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'Force_Plate')  # + ['Combined Moment'] + ['Combined CoP']
        self._force_plates[1] = ForcePlate.ForcePlate("Force_Plate_1", sensors["Force_Plate__Force_1"],
                                                      sensors["Force_Plate__Moment_1"])

        self._force_plates[2] = ForcePlate.ForcePlate("Force_Plate_2", sensors["Force_Plate__Force_2"],
                                                      sensors["Force_Plate__Moment_2"])

    def _make_EMGs(self):
        """
        generate EMG models
        :return: None
        """
        sensors = self.data_dict["Devices"]
        all_keys = self._filter_dict(sensors, 'EMG')
        T_EMG_keys = self._filter_dict(sensors, 'T_EMG')
        EMG_keys = [x for x in all_keys if x not in T_EMG_keys]
        for e_key, t_key in zip(EMG_keys, T_EMG_keys):
            self._T_EMGs[int(filter(str.isdigit, t_key))] = EMG.EMG(t_key, sensors[t_key])
            self._EMGs[int(filter(str.isdigit, e_key))] = EMG.EMG(e_key, sensors[e_key])
        # return dict([(key, sensors[key]) for key in keys])

    def _make_IMUs(self):
        """
        generate IMU models
        :return: None
        """
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'IMU')
        for key in keys:
            self._IMUs[int(filter(str.isdigit, key))] = IMU.IMU(key, sensors[key])

    def _make_Accelerometers(self):
        """
        generate the accel objects
        :return: None
        """
        sensors = self.data_dict["Devices"]
        keys = self._filter_dict(sensors, 'Accelerometers')
        for key in keys:
            self._accels[int(filter(str.isdigit, key))] = Accel.Accel(key, sensors[key])




if __name__ == '__main__':
    file = "/home/nathaniel/git/Gait_Analysis_Toolkit/Utilities/Walking01.csv"
    data = Vicon(file)
    print data.get_EMGs().keys()
