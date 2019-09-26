# import modules that I'm using
import matplotlib

matplotlib.use('TKAgg')
from PyQt5.uic import loadUiType
from PyQt5 import QtCore, QtWidgets
from matplotlib.figure import Figure

Ui_MainWindow, QMainWindow = loadUiType('/home/nathaniel/gait_analysis_toolkit/lib/Plotting_Tools/GUI/window.ui')

class TrialExaminer(QMainWindow, Ui_MainWindow ):

    def __init__(self):

        self.index = 0
        self.count = 0
        self.setupUi(self)
        self.objects = {}
        self.mplfigs.itemClicked.connect(self.changefig)
        fig = Figure()
        # self.canvas = FigureCanvas(fig)
        # self.toolbar = NavigationToolbar(self.canvas,
        #                                  self.mplwindow, coordinates=True)
        self.stacked_layout = QtWidgets.QStackedLayout(self.mplwindow)
        #self.stacked_layout.addWidget(self.toolbar)
        self.mplvl.addLayout(self.stacked_layout)

    def changefig(self, item):
        text = str(item.text())
        fig, index = self.objects[text]
        print index
        self.stacked_layout.setCurrentIndex(index)

    def addfig(self, fig):
        """

        :type fig: QT_Plotter.QT_Plotter
        """
        fig.initilize(self)
        widget = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout(widget)
        lay.addWidget(fig.toolbar)
        lay.addWidget(fig.canvas)
        self.stacked_layout.addWidget(widget)
        self.objects[fig.name] = (fig, self.index)
        self.index = self.index + 1
        self.mplfigs.addItem(fig.name)

    def update(self, data):
        """
        override method called when a message is passed
        :param data: sensors.
        :return: None
        """
        # loop through all the plots and update them
        for key, obj in self.objects.iteritems():
            obj[0].update()

    def start(self):
        """
        start the main loop of the GUI
        :return:
        """
        self.show()
        self.refesh()

