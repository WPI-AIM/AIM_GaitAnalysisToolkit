import numpy as np

from QT_Plotter import QT_Plotter


class FSR_BarGraph(QT_Plotter):

    def __init__(self, name, object, numbars=6):
        """

        :type object: List(FSR)
        """
        self.num_bars = np.arange(numbars)
        super(FSR_BarGraph, self).__init__(object, name)

    def initilize(self, parent):
        """
        Create the plot

        :param parent: parent window
        :return: None
        """

        self.ax.set_ylim([0, 1])
        self.bars = self.ax.bar(self.num_bars, [0] * len(self.num_bars), align='center', alpha=0.5)
        super(FSR_BarGraph, self).initilize(parent)

    def load(self, input):
        """Retrieve data from the input source and return an object."""
        return 1

    def update(self):
        """
        callback for to update the plot with new data
        :return:
        """
        data = []
        for sensor in self.object:
            data.append(sensor.get_values()[0])

        [rect.set_height(h) for rect, h in zip(self.bars, data)]
