import Queue

import numpy as np

from Sensors import Sensor
from QT_Plotter import QT_Plotter
import matplotlib.figure
import matplotlib.pyplot as pltlib

class Line_Graph(QT_Plotter):

    def __init__(self, name, object, num, labels):
        """

        :type object: Sensor.Sensor
        """
        self.num = num
        self.name = name
        self.labels = labels
        self.lines = []
        self.queue_size = 20
        self.ticks = 0
        self.queue = Queue.Queue(self.queue_size)
        super(Line_Graph, self).__init__(object, name)

    def initilize(self, parent):
        """
        Create the plot

        :param root: window to put the plot
        :param position: position in window to put the plot
        :return: None
        """
        print "init"
        for ii in xrange(self.num):
            line, = self.ax.plot([], [], self.colors[ii], lw=2)
            self.lines.append(line)

        self.ax.legend(self.labels, loc='upper left')

        super(Line_Graph, self).initilize(parent)

    def load(self, input):
        """Retrieve data from the input source and return an object."""
        return 1

    def update(self):
        """
        callback to update the plot
        :return: None
        """
        # read the sensor and put it into the queue
        values = self.object.get_values()

        if self.queue.qsize() >= self.queue_size:
            self.queue.get()

        self.queue.put(values)

        # get the x axis numbers
        # start it at 0 and go the the number of ticks then
        # once it reachs the queue size then contine the number of readings
        self.ticks = self.ticks + 1
        start = 0
        items = np.array(list(self.queue.queue))

        if self.ticks - self.queue_size > 0:
            start = self.ticks - self.queue_size

        x_data = range(start, self.ticks)

        # update the graph
        for ii, line in enumerate(self.lines):
            line.set_xdata(x_data)
            line.set_ydata(items[:, ii])
        # self.ax.relim()
        # self.ax.autoscale_view()
        # self.draw()


