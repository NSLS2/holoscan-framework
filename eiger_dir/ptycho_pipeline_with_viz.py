import numpy as np
from numpy.typing import NDArray
from eiger_connect_sample import EigerPtychoAppBase
from holoscan.core import Application, Operator, OperatorSpec
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
from nsls2ptycho.core.ptycho.utils import parse_config
from nsls2ptycho.core.ptycho_param import Param

gApp = None
simulate_position_data_stream = None

class OperatorWithQtSignal(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.qt_signal = None
        self.counter = 0
        
    def append_qt_signal(self, signal):
        self.qt_signal = signal
    

class PtychoDataViz(OperatorWithQtSignal):
    def setup(self, spec):
        spec.input("image")
        
    def compute(self, op_input, op_output, context):
        image = op_input.receive("image")
        if self.counter > 10:
            bla = image.copy()
            bla[207, 211] = 0
            bla[bla>500] = 500
            # emit signal
            self.qt_signal.emit(bla)
            self.counter = 0
        else:
            self.counter += 1

class PtychoPosViz(OperatorWithQtSignal):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []
    
    def setup(self, spec):
        spec.input("point")
        
    def compute(self, op_input, op_output, context):
        point = op_input.receive("point")
        self.data.append(point)
        if self.counter > 5:
            _data = np.array(self.data)
            # emit signal
            self.qt_signal.emit(_data)
            self.counter = 0
        else:
            self.counter += 1
            

class PtychoReconViz(OperatorWithQtSignal):
    def setup(self, spec):
        spec.input("input")
        
    def compute(self, op_input, op_output, context):
        image = op_input.receive("input")
        bla = np.abs(image.copy())[:, ::-1]
        bla_min, bla_max = np.percentile(bla[bla>0], 5), np.percentile(bla[bla>0], 95)
        bla = ((bla - bla_min)/ (bla_max - bla_min) + 0.5) / 2
        # emit signal
        self.qt_signal.emit(bla)


class EigerPtychoVizApp(EigerPtychoAppBase):
    def __init__(self, *args,
                 request_data_draw=None,
                 request_pos_draw=None,
                 request_recon_draw=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.request_data_draw = request_data_draw
        self.request_pos_draw = request_pos_draw
        self.request_recon_draw = request_recon_draw
    
    def compose(self):
        super().compose()
        data_viz = PtychoDataViz(self, name="data_viz")
        point_viz = PtychoPosViz(self, name="point_viz")
        recon_viz = PtychoReconViz(self, name="recon_viz")
        
        data_viz.append_qt_signal(self.request_data_draw)
        point_viz.append_qt_signal(self.request_pos_draw)
        recon_viz.append_qt_signal(self.request_recon_draw)
        
        self.add_flow(self._eiger_zmq_rx_pointer, data_viz, {("image", "image")})
        self.add_flow(self._pos_rx_pointer, point_viz, {("point", "point")})
        self.add_flow(self._recon_pointer, recon_viz)
        
    

class PtychoQtWorker(QtCore.QObject):
    request_data_draw = QtCore.pyqtSignal(np.ndarray)
    request_pos_draw = QtCore.pyqtSignal(np.ndarray)
    request_recon_draw = QtCore.pyqtSignal(np.ndarray)
    
    def run(self):
        """Run the Holoscan application."""
        # config_file = os.path.join(os.path.dirname(__file__), "config.yaml")
        
        eiger_ip = "0.0.0.0"
        eiger_port = "5555"
        msg_format = "cbor"
        simulate_position_data_stream = True
        position_data_path = f"/test_data/scan_257331.h5"
        recon_param = parse_config('/eiger_dir/ptycho_config',Param())
        recon_param.working_directory = "/eiger_dir/"
        recon_param.gpus = [0]
        # print(f"{recon_param.shm_name=}")
        recon_param.scan_num = 257331
        
        global gApp
        gApp = app = EigerPtychoVizApp(
            eiger_ip=eiger_ip,
            eiger_port=eiger_port,
            msg_format=msg_format,
            simulate_position_data_stream=simulate_position_data_stream,
            position_data_path=position_data_path,
            recon_param=recon_param,
            request_data_draw=self.request_data_draw,
            request_pos_draw=self.request_pos_draw,
            request_recon_draw=self.request_recon_draw)
        app.run()

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi()
        self.runHoloscanApp()
        
    def setupUi(self):
        self.setWindowTitle('Ptycho Data and Reconstruction View')
        self.resize(800, 1200)
        cw = QtWidgets.QWidget()
        self.setCentralWidget(cw)
        layout = QtWidgets.QGridLayout()
        cw.setLayout(layout)
        self.imv1 = pg.ImageView()
        self.pw1 = pg.PlotWidget()
        self.imv2 = pg.ImageView()
        
        layout.addWidget(self.imv1, 0, 0)
        layout.addWidget(self.pw1, 1, 0)
        layout.addWidget(self.imv2, 2, 0)
        
        layout.setRowMinimumHeight(0, 300)
        layout.setRowMinimumHeight(2, 300)
        
    def runHoloscanApp(self):
        """Run the Holoscan application in a separate thread."""
        self.thread = QtCore.QThread()
        self.worker = PtychoQtWorker()
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        # self.worker.finished.connect(self.thread.quit)
        # self.worker.finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        
        self.worker.request_data_draw.connect(self.data_draw)
        self.worker.request_pos_draw.connect(self.pos_draw)
        self.worker.request_recon_draw.connect(self.recon_draw)
        
        self.thread.start()
    
    def data_draw(self, data):
        self.imv1.setImage(data/250, autoHistogramRange=False, autoLevels=False)
        self.imv1.setHistogramRange(0, 500)
        self.imv1.setLevels(0, 1)
    
    def pos_draw(self, data):
        self.pw1.plot(data[:, 0], data[:, 1])

    def recon_draw(self, data):
        self.imv2.setImage(data, autoHistogramRange=False, autoLevels=False)
        self.imv2.setHistogramRange(-0.5, 1.5)
        self.imv2.setLevels(-0.5, 1.5)
        
if __name__ == "__main__":
    app = QtWidgets.QApplication([])
    main_window = MainWindow()
    main_window.show()
    app.exec_()
    

# def init_pg_app():

# app = pg.mkQApp()

# win = QtWidgets.QMainWindow()
# win.resize(800,1200)
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

# l.setRowMinimumHeight(0, 400)
# l.setRowMinimumHeight(1, 400)
# l.setRowMinimumHeight(2, 400)
# win.show()



# data_viz = PtychoDataViz(self, name="data_viz")
# point_viz = PtychoPosViz(self, name="point_viz")
# recon_viz = PtychoReconViz(self, name="recon_viz")
# self.add_flow(pos_rx, point_viz, {("point", "point")})
# self.add_flow(gather, sink, {("detmap", "detmap"), ("points", "points")})
# self.add_flow(eiger_zmq_rx, data_viz, {("image", "image")})
# self.add_flow(recon, recon_viz)

# import threading
#     def app_run():
#         global app
#         app.run()
#     thread = threading.Thread(target=app_run)
#     thread.start()
    

# print("I AM HERE")
    # pg.exec()
