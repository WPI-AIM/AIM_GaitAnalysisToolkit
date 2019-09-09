import matplotlib.image as mpimg

from lib.Plotting_Tools.QTPlotting import CoP_Plotter
from Sensors import Sensor


class CoP_Trial_Plotter(CoP_Plotter.CoP_Plotter):

    def __init__(self, name, left, right):
        """

        :type object: Sensor.Sensor
        """
        super(CoP_Plotter, self).__init__(object, name, left, right)

    def initilize(self, parent):
        """
        Create the plot

        :param root: window to put the plot
        :param position: position in window to put the plot
        :return: None
        """
        super(CoP_Plotter, self).initilize(parent=parent)

    def load(self, input):
        """Retrieve data from the input source and return an object."""
        return 1

    def update(self, data):
        """
        callback to update the plot
        :return: None
        """


        left = self.calc_CoP(self._left, self.left_locations)
        right = self.calc_CoP(self._right, self.right_locations)
        self.left.set_xdata([left[0]])
        self.left.set_ydata([left[1]])

        self.right.set_xdata([right[0]])
        self.right.set_ydata([right[1]])

    def calc_CoP(self, sensor, location):
        """
        calculate the CoP of the foot based on the FSR location
        and force
        CoP_x = sum_i(F_i * x_i)/sum_i(F_i)
        CoP_y = sum_i(F_i * y_i)/sum_i(F_i)
        :return:
        """
        fsrs = sensor

        total_force = 0
        centerX = 0
        centerY = 0

        for fsr, loc in zip(fsrs, location):
            total_force += fsr.get_values()[0]
            centerX += fsr.get_values()[0] * loc[0]
            centerY += fsr.get_values()[0] * loc[1]

        return [centerX / total_force, centerY / total_force]
