import sys
import time
import logging
import socket
import zmq
from argparse import ArgumentParser

from time import sleep

import numpy as np
import numpy.typing as npt
import cupy as cp
import json
import cbor2
import pprint
import traceback
import h5py

from dectris.compression import decompress

from scipy import signal as cpu
from cupyx.scipy import signal as gpu

from nsls2ptycho.core.ptycho.recon_ptycho_gui import ReconObject
from nsls2ptycho.core.ptycho.utils import parse_config
from nsls2ptycho.core.ptycho_param import Param

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op
# from holoscan.operators import HolovizOp

supported_encodings = {"bs32-lz4<": "bslz4", "lz4<": "lz4"}
supported_types = {"uint32": "uint32"}
def decode_json_message(zmq_message) -> tuple[str, npt.NDArray]:
    msg = json.loads(zmq_message.decode())

    # There should be more robust way to detect this frame
    if "htype" in msg and msg["htype"] == "dimage_d-1.0":
        data_encoding = msg.get("encoding", None)
        data_shape = msg.get("shape", None)
        data_type = msg.get("type", None)

        data_encoding_str = supported_encodings.get(data_encoding, None)
        if not data_encoding_str:
            raise RuntimeError(f"Encoding {data_encoding!r} is not supported")

        data_type_str = supported_types.get(data_type, None)
        if not data_type_str:
            raise RuntimeError(f"Encoding {data_type!r} is not supported")

        elem_type = getattr(np, data_type_str)
        elem_size = elem_type(0).nbytes
        decompressed = decompress(msg, data_encoding_str, elem_size=elem_size)
        image = np.frombuffer(decompressed, dtype=elem_type)
    else:
        msg_type = ""
        image = None
    return msg_type, image


tag_decoders = {
    69: "<u2",
    70: "<u4",
}
def decode_cbor_message(zmq_message) -> tuple[str, npt.NDArray]:
    msg = cbor2.loads(zmq_message)
    if msg["type"] == "image":
        msg_type = "image"

        # msg['series_id'] - these msgs have series_id
        # msg['image_id'] - and image ids

        msg_data = msg["data"]["threshold_1"]
        shape, contents = msg_data.value
        dtype = tag_decoders[contents.tag]

        if type(contents.value) is bytes:
            compression_type = None
            image = np.frombuffer(contents.value, dtype=dtype).reshape(shape)
        else:
            compression_type, elem_size, image = contents.value.value
            decompressed_bytes = decompress(image, compression_type, elem_size=elem_size)
            image = np.frombuffer(decompressed_bytes, dtype=dtype).reshape(shape)
    else:
        msg_type = ""
        image = None
    return msg_type, image


class EigerZmqRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("EigerZmqRxOp")
        logging.basicConfig(level=logging.INFO)

        self.endpoint = f"tcp://{eiger_ip}:{eiger_port}"
        self.msg_format = msg_format
        self.count = 0

        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)

        try:
            self.socket.connect(self.endpoint)
        except socket.error:
            self.logger.error("Failed to create socket")

    def setup(self, spec: OperatorSpec):
        spec.output("image")
        if simulate_position_data_stream:
            spec.output("count")


    def compute(self, op_input, op_output, context):
        while True:
            msg = self.socket.recv()
            try:
                if self.msg_format == "json":
                    msg_type, image_data = decode_json_message(msg)
                elif self.msg_format == "cbor":
                    msg_type, image_data = decode_cbor_message(msg)

                if msg_type == "image":
                    self.count += 1
                    self.logger.info(f"Successfully processed {self.count} frames")
                    op_output.emit(image_data, "image")
                    if simulate_position_data_stream:
                        op_output.emit(self.count, "count") # emit count to trigger the position transmitter to emit corresponding point
                else: # probably should have a better handling of start/end messages
                    self.logger.info("-" * 80)

                    if self.count == 0:
                        self.logger.info("Image series start")
                    else:
                        self.logger.info("Image series end")
                        self.count = 0
                    self.logger.info("-" * 80)

            except Exception as ex:
                result = "ERROR: Failed to process message: {ex}"
                print(f"{pprint.pformat(result)}")
                print(traceback.format_exc())

            return


class PositionSimTxOp(Operator):
    '''
    Simulator of position data stream. Receives a signal from EigerZmqExOp and emits corresponding point data.
    '''
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PositionSimTxOp")
        logging.basicConfig(level=logging.INFO)
        with h5py.File(position_data_path) as f:
            self.points = f["points"][()].T
            self.N = self.points.shape[0]
            self.index = 0

    def setup(self, spec: OperatorSpec):
        spec.input("index")
        spec.output("point")

    def compute(self, op_input, op_output, context):
        op_input.receive("index")
        if self.index < self.N:
            self.logger.info(f"Emitting {self.index} point coordinate")
            point = self.points[self.index, :]
            op_output.emit(point, "point")
            self.index += 1

class PositionRxOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PositionRxOp")
        logging.basicConfig(level=logging.INFO)
        # some logic that goes into setting up the position data receiver (ZMQ, UDP, etc)

    def setup(self, spec: OperatorSpec):
        if simulate_position_data_stream:
            spec.input("point_input")
        spec.output("point")

    def compute(self, op_input, op_output, context):
        if simulate_position_data_stream:
            data = op_input.receive("point_input")
        else:
            data = np.array([0, 0]) # placeholder - this should be changed to something that will actually receive the data
        self.logger.info(f"Emitting point data {data}")
        op_output.emit(data, "point")


class GatherOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("GatherOp")
        logging.basicConfig(level=logging.INFO)
        self.image_list = []
        self.point_list = []
        self.counter = 0
        self.batchsize = 200

    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.input("point")
        spec.output("detmap")
        spec.output("points")
        
    def compute(self, op_input, op_output, context):
        image = op_input.receive("image")
        point = op_input.receive("point")
        
        self.image_list.append(image)
        self.point_list.append(point)
        
        if self.counter < (self.batchsize - 1):
            self.counter += 1
        else:
            detmap = np.array(self.image_list)
            points = np.array(self.point_list)
            op_output.emit(detmap, "detmap")
            op_output.emit(points, "points")
            self.counter = 0
            self.logger.info("Emitting data batch")
            

class ReconOp(Operator):
    def __init__(self, *args, param=None, postprocessing_flag=False, **kwargs):
        self.logger = logging.getLogger("ReconOp")
        logging.basicConfig(level=logging.INFO)
        super().__init__(*args, **kwargs)
        self.recon_obj = ReconObject(param=param, postprocessing_flag=postprocessing_flag)
        
    def setup(self, spec):
        spec.input("detmap")
        spec.input("points")
        spec.output("result")
    
    def compute(self, op_input, op_output, context):
        detmap = op_input.receive("detmap")
        points = op_input.receive("points")
        
        self.logger.info("Reconstruction started")
        nz = detmap.shape[0]
        ic = np.ones(points.shape[0])
        
        self.recon_obj.set_data_arrays(detmap, points.T, ic, nz)
        self.recon_obj.recon_ptycho()
        self.logger.info("Reconstruction finished")
        
        op_output.emit(self.recon_obj.recon.obj_ave, "result")
        # op_output.emit(1, "result")
    
    def stop(self, *args, **kwargs):
        self.recon_obj.finalize()
        super().stop(*args, **kwargs)



# def init_pg_app():
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtWidgets
app = pg.mkQApp()

win = QtWidgets.QMainWindow()
win.resize(800,1200)
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

l.setRowMinimumHeight(0, 400)
l.setRowMinimumHeight(1, 400)
l.setRowMinimumHeight(2, 400)
win.show()

class PtychoDataViz(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.counter = 0
    
    def setup(self, spec):
        spec.input("image")
        
    def compute(self, op_input, op_output, context):
        image = op_input.receive("image")
        if self.counter > 10:
            bla = image.copy()
            bla[207, 211] = 0
            # bla[bla>500] = 500
            global imv1
            imv1.setImage(bla/250, autoHistogramRange=False, autoLevels=False)
            # imv1.setHistogramRange(0, 500)
            imv1.setLevels(0, 1)
            self.counter = 0
        else:
            self.counter += 1

class PtychoPosViz(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = []
        self.counter = 0
    
    def setup(self, spec):
        spec.input("point")
        
    def compute(self, op_input, op_output, context):
        global pw1
        point = op_input.receive("point")
        self.data.append(point)
        if self.counter>10:
            _data = np.array(self.data)
            pw1.plot(_data[:, 0], _data[:, 1])
            self.counter = 0
        else:
            self.counter += 1
            

class PtychoReconViz(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    
    def setup(self, spec):
        spec.input("input")
        
    def compute(self, op_input, op_output, context):
        global imv2
        image = op_input.receive("input")
        bla = np.abs(image.copy())[:, ::-1]
        bla_min, bla_max = np.percentile(bla[bla>0], 5), np.percentile(bla[bla>0], 95)
        bla = ((bla - bla_min)/ (bla_max - bla_min) + 0.5) / 2
        
        # bla[207, 211] = 0
        # bla[bla>500] = 500
        # imv2.setImage(bla, autoHistogramRange=False, autoLevels=False)
        imv2.setImage(bla, autoHistogramRange=False, autoLevels=False)
        imv2.setHistogramRange(-0.5, 1.5)
        imv2.setLevels(-0.5, 1.5)


# @create_op(inputs=("image", "point"))
# def sink_func(image, point):
#     print(f"SinkOp received image: shape={image.shape}, total_counts={np.sum(image)}")
#     print(f"SinkOp received point coordinates: {point=}")

# @create_op(inputs=("detmap", "points"))
# def sink_func(detmap, points):
#     print(f"SinkOp received image: shape={detmap.shape}, total_counts={np.sum(detmap, axis=(1,2))}")
#     print(f"SinkOp received points: shape={points.shape}")

@create_op(inputs="result")
def sink_func(result):
    print(f"SinkOp received the result from reconstruction {result=}")


class EigerPtychoApp(Application):
    def compose(self):
        global simulate_position_data_stream, recon_param
        
        eiger_zmq_rx = EigerZmqRxOp(self, name="eiger_zmq_rx")
        pos_rx = PositionRxOp(self, name="pos_rx")
        gather = GatherOp(self, name="gather")
        recon = ReconOp(self, name="recon", param=recon_param, postprocessing_flag=False)
        # sink = sink_func(self, name="sink")
        data_viz = PtychoDataViz(self, name="data_viz")
        point_viz = PtychoPosViz(self, name="point_viz")
        recon_viz = PtychoReconViz(self, name="recon_viz")
        
        self.add_flow(eiger_zmq_rx, gather, {("image", "image")})
        self.add_flow(eiger_zmq_rx, data_viz, {("image", "image")})
        self.add_flow(pos_rx, gather, {("point", "point")})
        self.add_flow(pos_rx, point_viz, {("point", "point")})
        # self.add_flow(gather, sink, {("detmap", "detmap"), ("points", "points")})
        self.add_flow(gather, recon, {("detmap", "detmap"), ("points", "points")})
        # self.add_flow(recon, sink)
        self.add_flow(recon, recon_viz)
        
        if simulate_position_data_stream:
            pos_sim_tx = PositionSimTxOp(self, name="pos_sim_tx")
            self.add_flow(eiger_zmq_rx, pos_sim_tx, {("count", "index")})
            self.add_flow(pos_sim_tx, pos_rx)



if __name__ == "__main__":
    parser = ArgumentParser(description="Eiger ingest example")
    parser.add_argument(
        "--eiger_ip",
        type=str,
        default=...,
        help=(
            "Eiger Detector IP address"
        ),
    )
    parser.add_argument(
        "--eiger_port",
        type=str,
        default="9999",
        help=("Eiger Detector port"),
    )
    parser.add_argument(
        "-m",
        "--message_format",
        type=str,
        choices=["json", "cbor"],
        default="json",
        help=("Eiger message format"),
    )
    parser.add_argument(
        "-p",
        "--position_data_source",
        type=str,
        default="scan_257331.h5",
        help=("Position data source"),
    )

    args = parser.parse_args()
    eiger_ip = args.eiger_ip
    eiger_port = args.eiger_port
    msg_format = args.message_format
    if args.position_data_source == "stream":
        simulate_position_data_stream = False
        position_data_path = None
    elif args.position_data_source.endswith(".h5"):
        simulate_position_data_stream = True
        position_data_path = f"/test_data/{args.position_data_source}"
    
    recon_param = parse_config('/eiger_dir/ptycho_config',Param())
    recon_param.working_directory = "/eiger_dir/"
    recon_param.gpus = [0]
    # print(f"{recon_param.shm_name=}")
    recon_param.scan_num = 257331

    app = EigerPtychoApp()
    
    import threading
    def app_run():
        global app
        app.run()
    thread = threading.Thread(target=app_run)
    thread.start()
    
    print("I AM HERE")
    pg.exec()
