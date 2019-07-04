import csv


def open_vicon_file(file_path):
    """
    parses the Vicon sensor data into a dictionary
    :param file_path: file path
    :return: dictionary of the sensors
    :rtype: dict
    """
    indices = {}
    data = {}
    current_name = None
    last_frame = None

    # open the file and get the column names, axis, and units
    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        raw_data = list(reader)
        column_names = raw_data[4]
        axis = raw_data[5]
        unit = raw_data[6]

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
            data[current_name][dir]["data"] = {}
            data[current_name][dir]["unit"] = unit[index]

    # Put all the data in the correct sub dictionary.

    for row in raw_data[7:]:
        frame = int(row[0])

        for key, value in data.iteritems():
            for sub_key, sub_value in value.iteritems():

                if frame not in sub_value["data"]:
                    sub_value["data"][frame] = []
                index = indices[(key, sub_key)]
                sub_value["data"][int(frame)].append(float(row[index]))
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
    data = open_vicon_file(file)
    print data["Imported Delsys Trigno IMU EMG 2.0 #4 - Sensor 8"]["IM EMG8"]
