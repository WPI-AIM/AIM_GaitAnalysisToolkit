import csv
import numpy as np

def open_vicon_file(file_path):
    raw_data = []
    column_names = []
    axis = []
    unit = []
    with open(file_path, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        raw_data = list(reader)
        column_names = raw_data[4]
        axis = raw_data[5]
        unit = raw_data[6]

    data = {}
    print len(axis)
    print len(column_names)
    print len(unit)
    current_name = None

    for index, name in enumerate(column_names):

        if index <= 1:
            continue
        else:
            if len(name) > 0:
                current_name = name
                data[current_name] = {}
            dir = axis[index]
            data[current_name][dir] = {}
            data[current_name][dir]["data"] = []
            data[current_name][dir]["unit"] = unit[index]

    print data["Combined CoP"]

def column(matrix, i):
    return [row[i] for row in matrix]

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
    open_vicon_file(file)