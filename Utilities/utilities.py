import csv


def open_vicon_file(file_path, output_names):
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
    segs = _seperate_csv_sections(raw_data)
    for index, output in enumerate(output_names):
        data[output] = _extract_values(raw_data, segs[index], segs[index + 1])
    return data


def _seperate_csv_sections(all_data):

    col1 = [row[0] for row in all_data]
    devices = col1.index("Devices")
    joints = col1.index("Joints")
    model_output = col1.index("Model Outputs")
    segments = col1.index("Segments")
    trajs = col1.index("Trajectories")
    segs = [devices, joints, model_output, segments, trajs, len(col1)]
    return segs


def _fix_col_names(names):
    fixed_names = []
    get_index = lambda x: x.index("Sensor") + 7
    for name in names:  # type: str

        if "Subject" in name:
            fixed = ''.join(
                [i for i in name.replace("Subject", "").replace(":", "").replace("|", "") if not i.isdigit()]).strip()
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

def _extract_values(raw_data, start, end):
    indices = {}
    data = {}
    current_name = None
    last_frame = None

    column_names = _fix_col_names(raw_data[start + 2])
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

def open_exo_file(file_path):
    '''

    :param file_path: path to the data file
    :return: values of the sensors
    :rtype: dict
    '''
    data = {}
    with open(file_path, mode='r') as csv_file:

        csv_reader = csv.DictReader(csv_file)
        keys = csv_reader.fieldnames
        for key in keys:
            data[key] = []
        for row in csv_reader:
            for key in keys:
                data[key].append([float(x.strip()) for x in row[key].split(',')])

    return data


if __name__ == '__main__':
    # file = "/home/nathaniel/git/exoserver/Main/subject_37_trial_1.csv"
    # print open_exo_file(file)["Pot_Left_Ankle"]
    file = "Walking01.csv"
    data = open_vicon_file(file, ["Devices", "Joints", "Model Outputs", "Segments", "Trajectories"])
    print data["Devices"].keys()
