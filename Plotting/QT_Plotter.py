import abc
from matplotlib.backends.backend_qt5agg import (
    FigureCanvas, NavigationToolbar2QT as NavigationToolbar)
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class QT_Plotter(object):

    fig = None  # type: Figure

    def __init__(self, object, name):
        """

        :type object: Sensor.Sensor
        """
        self.object = object
        self.name = name
        self.fig = Figure(figsize=(3, 2), dpi=100)
        self.canvas = FigureCanvas(self.fig)
        self.toolbar = None
        self.ax = self.fig.add_subplot(111)
        self.colors = ['r-', 'g-', 'b-', 'k-', 'm-', 'c-', 'y-']

    @abc.abstractmethod
    def initilize(self, parent):
        """
        This function is used to set up the windows
        set up a window to plot
        :param parent: window to put the plot in
        :return: None
        """
        #
        # FigureCanvas.__init__(self, self.fig)
        # self.setParent(parent)
        # FigureCanvas.setSizePolicy(self,
        #                            QtGui.QSizePolicy.Expanding,
        #                            QtGui.QSizePolicy.Expanding)
        # FigureCanvas.updateGeometry(self)

        self.toolbar = NavigationToolbar(self.canvas, parent)
        self.canvas.setParent(parent)
        self.set_title(self.name)
        self.set_axis_names()

        # self.set_fitler_menu()
        return

    @abc.abstractmethod
    def update(self):
        """
        override function to be called to update the plot
        :return:
        """
        pass


    def set_title(self, title="some_graph"):
        """
        set the title of the plot
        :param title: name of the plot
        :type: str
        :return:
        """
        self.ax.set_title(title)
        return

    def set_axis_names(self, x="x", y="y"):
        """
        set the axis names
        :param x: name of x axis
        :param y: name of y axis
        :type x: str
        :type y: str
        :return:
        """
        self.ax.set_xlabel(x)
        self.ax.set_ylabel(y)

    def flush(self):
        """
        Update the plot graphics
        :return:
        """
        # update the axis limits

        #redraw

        self.ax.relim()
        self.ax.autoscale_view()
        self.canvas.draw()

