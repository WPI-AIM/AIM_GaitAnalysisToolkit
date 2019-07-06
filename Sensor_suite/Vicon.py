import csv

import Accel
import EMG
import ForcePlate
import IMU


class Vicon(object):

    def __init__(self, file_path):
        self._file_path = file_path
        self.output_names = ["Devices", "Joints", "Model Outputs", "Segments", "Trajectories"]
        self._number_of_frames = 0
        self._T_EMGs = {}
        self._EMGs = {}
        self._force_plates = {}
        self._IMUs = {}
        self._accels = {}
        self.data_dict = self.open_vicon_file(self._file_path, self.output_names)
        self._make_Accelerometers()
        self._make_EMGs()
        self._make_force_plates()
        self._make_IMUs()

    def _find_number_of_frames(self, col):
        """
        Finds the number and sets of frames
        :param col: column to search in
        :return: None
        """
        index = col.index("Frame") + 2
        current_number = col[index]

        while current_number.isdigit():
            index += 1
            current_number = col[index]

        self.number_of_frames = col[index - 1]

    @property
    def number_of_frames(self):
        """

        :return: number of frames
        :rtype: int
        """
        return self._number_of_frames

    @number_of_frames.setter
    def number_of_frames(self, value):
        """

        :param value:
        :return:
        """
        self._number_of_frames = value

    @property
    def accels(self):
        """
        Get the Accels dict
        :return: Accels
        :type: dict
        """
        return self._accels

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
        keys = self._filter_dict(sensors, 'Accel')
        print keys
        for key in keys:
            self._accels[int(filter(str.isdigit, key))] = Accel.Accel(key, sensors[key])

    def open_vicon_file(self, file_path, output_names):
        """
        parses the Vicon sensor data into a dictionary
        :param file_path: file path
        :return: dictionary of the sensors
        :rtype: dict
        """
        # open the file and get the column names, axis, and units
        with open(file_path, mode='r') as csv_file:
            reader = csv.reader(csv_file)
            raw_data = list(reader)

        # output_names = ["Devices", "Joints", "Model Outputs", "Segments", "Trajectories"]
        data = {}
        segs = self._seperate_csv_sections(raw_data)
        for index, output in enumerate(output_names):
            data[output] = self._extract_values(raw_data, segs[index], segs[index + 1])
        return data

    def _seperate_csv_sections(self, all_data):

        col1 = [row[0] for row in all_data]
        devices = col1.index("Devices")
        joints = col1.index("Joints")
        model_output = col1.index("Model Outputs")
        segments = col1.index("Segments")
        trajs = col1.index("Trajectories")
        self._find_number_of_frames(col1)
        segs = [devices, joints, model_output, segments, trajs, len(col1)]
        return segs

    def _fix_col_names(self, names):
        fixed_names = []
        get_index = lambda x: x.index("Sensor") + 7
        for name in names:  # type: str

            if "Subject" in name:
                fixed = ''.join(
                    [i for i in name.replace("Subject", "").replace(":", "").replace("|", "") if
                     not i.isdigit()]).strip()
                fixed_names.append(fixed)
                continue

            elif "AMTI" in name:

                if "Force" in name:
                    unit = "_Force_"
                elif "Moment" in name:
                    unit = "_Moment_"
                elif "CoP" in name:
                    unit = "_CoP_"

                number = name[name.find('#') + 1]
                fixed = "Force_Plate_" + unit + str(number)
                fixed_names.append(fixed)

            elif "Trigno EMG" in name:
                fixed = "T_EMG_" + name[-1]
                fixed_names.append(fixed)

            elif "Accelerometers" in name:
                fixed = "Accel_" + name[get_index(name):]
                fixed_names.append(fixed)

            elif "IMU AUX" in name:
                fixed = "IMU_" + name[get_index(name):]
                fixed_names.append(fixed)

            elif "IMU EMG" in name:
                fixed = "EMG_" + name[get_index(name):]
                fixed_names.append(fixed)
            else:
                fixed_names.append("")

        return fixed_names

    def _extract_values(self, raw_data, start, end):
        indices = {}
        data = {}
        current_name = None
        last_frame = None

        column_names = self._fix_col_names(raw_data[start + 2])
        # column_names = raw_data[start + 2]
        remove_numbers = lambda str: ''.join([i for i in str if not i.isdigit()])
        axis = map(remove_numbers, raw_data[start + 3])
        unit = raw_data[start + 4]

        # Build the dict to store everything
        for index, name in enumerate(column_names):

            if index <= 1:
                continue
            else:
                if len(name) > 0:
                    current_name = name
                    data[current_name] = {}
                dir = axis[index]
                indices[(current_name, dir)] = index
                data[current_name][dir] = {}
                data[current_name][dir]["data"] = []
                data[current_name][dir]["unit"] = unit[index]

        # Put all the data in the correct sub dictionary.

        for row in raw_data[start + 5:end - 1]:

            frame = int(row[0])
            for key, value in data.iteritems():
                for sub_key, sub_value in value.iteritems():
                    index = indices[(key, sub_key)]
                    if row[index] is '':
                        val = 0
                    else:
                        val = float(row[index])
                    sub_value["data"].append(val)
        return data




if __name__ == '__main__':
    file = "/home/nathaniel/git/Gait_Analysis_Toolkit/Utilities/Walking01.csv"
    data = Vicon(file)
    print data.get_accel(1).name
