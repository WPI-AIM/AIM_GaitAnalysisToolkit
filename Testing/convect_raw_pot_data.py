import csv
import matplotlib.pyplot as plt


def plot_data(hip, knee):

    hip_index = (1000, 3000 )
    knee_index = ( 3500, 3500)
    hip_offset = 0
    knee_offset = 0
    convect = 310.0/1023
    hip_angle = []
    knee_angle = []

    for i in range(len(hip)):
        hip_angle.append( (hip[i] - hip_offset)* convect  )
        knee_angle.append((knee[i] - knee_offset)*convect)

    fig = plt.figure()
    plt.plot(hip_angle[hip_index[0]:hip_index[1] ])
    plt.show()



if __name__ == "__main__":
    file = "/home/nathanielgoldfarb/Downloads/exoskele_rom.csv"
    hip = []
    knee = []
    with open(file, mode='r') as csv_file:
        reader = csv.reader(csv_file)
        for line in reader:
            hip.append(int(line[0]))
            knee.append(int(line[1]))
    print hip
    plot_data(hip, knee)
