import matplotlib.pyplot as plt
from matplotlib import style
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
)
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar,
)
from matplotlib.figure import Figure
from qtpy.QtWidgets import (
    QComboBox,
    QHBoxLayout,
    QPushButton,
    QSizePolicy,
    QSpacerItem,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)


class MplCanvas(FigureCanvas):
    def __init__(self, **kwargs):
        # use 'ggplot' styling
        with style.context("ggplot"):
            plt.rcParams["font.size"] = 8
            plt.rcParams["axes.titlesize"] = 10
            plt.rcParams["axes.labelsize"] = 8
            plt.rcParams["xtick.labelsize"] = 7
            plt.rcParams["ytick.labelsize"] = 7
            plt.rcParams["legend.fontsize"] = 7
            self.fig = Figure(**kwargs)
            self.ax = self.fig.add_subplot(111)
            FigureCanvas.__init__(self, self.fig)


class MatplotlibWidget(QWidget):
    """a small convenient matplotlib widget with navigation toolbar on top"""

    def __init__(self, **kwargs):
        """
        See `matplotlib.figure.Figure` documentation for keyworded parameters
        """
        super().__init__()
        self.setMinimumSize(300, 250)
        self.canvas = MplCanvas(**kwargs)
        self.toolbar = NavigationToolbar(self.canvas, self)
        # style the toolbar background
        self.toolbar.setStyleSheet("background-color: #e5e5e5")
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.toolbar)
        self.layout.addWidget(self.canvas)
        self.setLayout(self.layout)


class StackedPlotWidget(QWidget):
    def __init__(self):
        super().__init__()

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(2)

        # Initialize the QComboBox for plot selection
        self.plotSelector = QComboBox()
        self.plotSelector.currentIndexChanged.connect(self.change_plot)
        self.layout.addWidget(self.plotSelector)

        # add navigation buttons
        self.setupNavigationButtons()

        # Initialize the QStackedWidget for displaying plots
        self.stackedWidget = QStackedWidget()
        self.layout.addWidget(self.stackedWidget)

        # Store references to the plots' axes to return for custom plotting
        self.mplwidgets = {}

        # connect to combo box signal
        self.stackedWidget.currentChanged.connect(self.update_plot_selector)

    def update_plot_selector(self, index):
        self.plotSelector.setCurrentIndex(index)

    def setupNavigationButtons(self):
        self.prevButton = QPushButton("<")
        self.nextButton = QPushButton(">")
        self.prevButton.setFixedSize(30, 30)
        self.nextButton.setFixedSize(30, 30)

        # connect button to slots
        self.prevButton.clicked.connect(self.prev_plot)
        self.nextButton.clicked.connect(self.next_plot)

        buttonLayout = QHBoxLayout()

        spacer = QSpacerItem(
            30, 20, QSizePolicy.Expanding, QSizePolicy.Minimum
        )
        buttonLayout.addItem(spacer)
        buttonLayout.addWidget(self.prevButton)
        buttonLayout.addWidget(self.nextButton)

        buttonContainer = QWidget()
        buttonContainer.setLayout(buttonLayout)

        self.layout.addWidget(buttonContainer)

    def add_subplot(self, name="Plot", figsize=(4, 3)):
        """
        Adds a new plot to the QStackedWidget and an entry to the QComboBox.
        Returns a matplotlib `Axes` object for plotting.
        """

        # if there's already an 'Axes' with a given name
        if name in self.mplwidgets.keys():
            self.stackedWidget.removeWidget(self.mplwidgets[name])

            plotWidget = MatplotlibWidget(figsize=figsize)

            self.stackedWidget.addWidget(plotWidget)
            self.mplwidgets[name] = plotWidget
            # choose the newly added plot
            self.stackedWidget.setCurrentIndex(self.stackedWidget.count() - 1)
            return plotWidget.canvas.ax

        else:
            # otherwise, create a new QWidget for the plot
            plotWidget = MatplotlibWidget(figsize=figsize)

            self.stackedWidget.addWidget(plotWidget)
            self.plotSelector.addItem(name)

            # store reference to Axes
            self.mplwidgets[name] = plotWidget
            self.stackedWidget.setCurrentIndex(self.stackedWidget.count() - 1)

            # return new 'Axes'
            return plotWidget.canvas.ax

    def change_plot(self, index):
        """Change the current plot displayed in the QStackedWidget based on the
        selected index.

        """
        self.stackedWidget.setCurrentIndex(index)

    def prev_plot(self):
        # Go to the previous plot
        currentIndex = self.plotSelector.currentIndex()
        if currentIndex > 0:
            self.plotSelector.setCurrentIndex(currentIndex - 1)

    def next_plot(self):
        # Go to the next plot
        currentIndex = self.plotSelector.currentIndex()
        if currentIndex < self.plotSelector.count() - 1:
            self.plotSelector.setCurrentIndex(currentIndex + 1)

    def clear(self):
        """
        Clears all plots from the widget.
        """
        # Clear the QComboBox
        self.plotSelector.clear()

        # Remove all widgets from the QStackedWidget
        while self.stackedWidget.count():
            widget = self.stackedWidget.widget(0)
            self.stackedWidget.removeWidget(widget)
            widget.deleteLater()

        # Clear the axes list
        self.mplwidgets = {}


if __name__ == "__main__":

    import sys

    import numpy as np
    from qtpy.QtWidgets import QApplication

    app = QApplication(sys.argv)
    w = StackedPlotWidget()

    x = np.linspace(0, 10, num=50)
    ax1 = w.add_plot(name="plot #1")
    ax1.plot(x, np.sin(x), "k-", lw=2)

    ax2 = w.add_plot(name="plot #2")
    ax2.plot(x, np.cos(x), "b-", lw=2)

    w.show()
    sys.exit(app.exec_())
