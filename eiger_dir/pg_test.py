
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
app = pg.mkQApp()

win = QtWidgets.QMainWindow()
win.resize(800,900)
win.setWindowTitle('Ptycho Data and Reconstruction View')
cw = QtWidgets.QWidget()
win.setCentralWidget(cw)
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
win.show()

pw1.plot([0,1,2,3], [0,1,2,1])

if __name__ == '__main__':
    pg.exec()
