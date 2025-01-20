
# import numpy as np

# import pyqtgraph as pg
# from pyqtgraph.Qt import QtCore

# app = pg.mkQApp("Plotting Example")
# #mw = QtWidgets.QMainWindow()
# #mw.resize(800,800)

# win = pg.GraphicsLayoutWidget(show=True, title="Basic plotting examples")
# win.resize(1000,600)
# win.setWindowTitle('pyqtgraph example: Plotting')

# # Enable antialiasing for prettier plots
# pg.setConfigOptions(antialias=True)

# p1 = win.addPlot(title="Basic array plotting", y=np.random.normal(size=100))

# if __name__ == '__main__':
#     pg.exec()


import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
import numpy as np

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Ptycho Data and Reconstruction View')
        self.resize(800, 1200)
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        l = QtWidgets.QGridLayout()
        cw.setLayout(l)
        imv1 = pg.ImageView()
        pw1 = pg.PlotWidget()
        imv2 = pg.ImageView()
        l.addWidget(imv1, 0, 0)
        l.addWidget(pw1, 1, 0)
        # ppos = l.addPlot(title='positions')
        # curve = ppos.plot()
        l.addWidget(imv2, 2, 0)
        l.setRowMinimumHeight(0, 300)
        l.setRowMinimumHeight(2, 300)
        # win.show()
        
        imv2.ui.histogram.hide()
        imv2.ui.roiBtn.hide()
        imv2.ui.menuBtn.hide()
        imv2.setColorMap(pg.colormap.get('viridis'))
        
        bla = pw1.plot([0], [0])
        bla.setData([0,1,2,3], [10,11,12,11])
        
        
        np.random.seed(1)
        data = np.random.randn(698, 698)/3
        imv2.setImage(data, autoHistogramRange=False, autoLevels=False)
        imv2.setHistogramRange(-1.0, 1.0)
        imv2.setLevels(-0.7, 0.7)
        
        
if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()

# app = pg.mkQApp()

# win = QtWidgets.QMainWindow()
# win.resize(800,900)
# win.setWindowTitle('Ptycho Data and Reconstruction View')
# cw = QtWidgets.QWidget()
# win.setCentralWidget(cw)
# l = QtWidgets.QGridLayout()


# cw.setLayout(l)
# imv1 = pg.ImageView()
# pw1 = pg.PlotWidget()
# imv2 = pg.ImageView()
# l.addWidget(imv1, 0, 0)
# l.addWidget(pw1, 1, 0)
# # ppos = l.addPlot(title='positions')
# # curve = ppos.plot()
# l.addWidget(imv2, 2, 0)

# l.setRowMinimumHeight(0, 300)
# l.setRowMinimumHeight(2, 300)
# win.show()

# pw1.plot([0,1,2,3], [0,1,2,1])

# if __name__ == '__main__':
#     pg.exec()
