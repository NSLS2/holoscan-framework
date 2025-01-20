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

from nsls2ptycho.core.ptycho.recon_ptycho_gui import create_recon_object, deal_with_init_prb
from nsls2ptycho.core.ptycho.utils import parse_config
from nsls2ptycho.core.ptycho_param import Param

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op
# from holoscan.operators import HolovizOp

simulate_position_data_stream = None
eiger_ip = None
eiger_port = None
msg_format = None

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
    def __init__(self, fragment, *args,
                 eiger_ip:str=None,
                 eiger_port:str=None,
                 msg_format:str=None,
                 simulate_position_data_stream:bool=False,
                 **kwargs):
        
        self.endpoint = f"tcp://{eiger_ip}:{eiger_port}"
        self.msg_format = msg_format
        self.count = 0
        self.simulate_position_data_stream = simulate_position_data_stream
        context = zmq.Context()
        self.socket = context.socket(zmq.PULL)

        try:
            self.socket.connect(self.endpoint)
        except socket.error:
            self.logger.error("Failed to create socket")
        
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("EigerZmqRxOp")
        logging.basicConfig(level=logging.INFO)

    def setup(self, spec: OperatorSpec):
        spec.output("image")
        if self.simulate_position_data_stream:
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
                    # self.logger.info(f"Successfully processed {self.count} frames")
                    op_output.emit(image_data, "image")
                    if self.simulate_position_data_stream:
                        op_output.emit(self.count, "count") # emit count to trigger the position transmitter to emit corresponding point
                else: # probably should have a better handling of start/end messages
                    pass
                    # self.logger.info("-" * 80)

                    # if self.count == 0:
                    #     # self.logger.info("Image series start")
                    # else:
                    #     # self.logger.info("Image series end")
                    #     self.count = 0
                    # # self.logger.info("-" * 80)

            except Exception as ex:
                result = "ERROR: Failed to process message: {ex}"
                print(f"{pprint.pformat(result)}")
                print(traceback.format_exc())

            return


class PositionSimTxOp(Operator):
    '''
    Simulator of position data stream. Receives a signal from EigerZmqExOp and emits corresponding point data.
    '''
    def __init__(self, *args, position_data_path:str=None, **kwargs):
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
            # self.logger.info(f"Emitting {self.index} point coordinate")
            point = self.points[self.index, :]
            op_output.emit(point, "point")
            self.index += 1

class PositionRxOp(Operator):
    def __init__(self, *args,
                 simulate_position_data_stream:bool=None,
                 **kwargs):
        self.simulate_position_data_stream = simulate_position_data_stream
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PositionRxOp")
        logging.basicConfig(level=logging.INFO)
        # some logic that goes into setting up the position data receiver (ZMQ, UDP, etc)

    def setup(self, spec: OperatorSpec):
        if self.simulate_position_data_stream:
            spec.input("point_input")
        spec.output("point")

    def compute(self, op_input, op_output, context):
        if self.simulate_position_data_stream:
            data = op_input.receive("point_input")
        else:
            data = np.array([0, 0]) # placeholder - this should be changed to something that will actually receive the data
        # self.logger.info(f"Emitting point data {data}")
        op_output.emit(data, "point")


class ImagePreprocessOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("ImagePreprocessOp")
        logging.basicConfig(level=logging.INFO)
        self.roi = np.array([[1, 257], [1,257]])
        self.detmap_threshold = 0
        self.badpixels = np.array([[207], [211]])
        
    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.output("diff_amp")
        
    def compute(self, op_input, op_output, context):
        _image = op_input.receive("image")
        image = _image.copy()
        for bd in self.badpixels.T:
            x = int(bd[0])
            y = int(bd[1])
            image[x,y] = np.median(image[x-1:x+2,y-1:y+2])
            
        image = image[self.roi[0,0]:self.roi[0,1], self.roi[1,0]:self.roi[1,1]]
        image = np.rot90(image,axes=(1,0))
        image = np.fft.fftshift(image,axes=(1,0))
        if self.detmap_threshold > 0:
            image[image<self.detmap_threshold] = 0
        # diff_l = np.zeros_like(image, dtype=float_precision)
        diff_amp = np.sqrt(image)
        op_output.emit(diff_amp, "diff_amp")
        

class PositionPreprocessOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PositionPreprocessOp")
        logging.basicConfig(level=logging.INFO)
        self.point_base = None
        
    def setup(self, spec: OperatorSpec):
        spec.input("point")
        spec.output("point_rel")
        
    def compute(self, op_input, op_output, context):
        point = op_input.receive("point")
        
        # if self.point_base is None:
            # self.point_base = point
        
        # op_output.emit(point - self.point_base, "point_rel")
        op_output.emit(point, "point_rel")
        

class GatherOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("GatherOp")
        logging.basicConfig(level=logging.INFO)
        self.point_list = []
        self.counter = 0
        self.batchsize = 500
        
        self.diff_d_to_add = np.zeros((self.batchsize, 256, 256))
        self.points_to_add = np.zeros((2, self.batchsize))

    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.input("point")
        spec.output("detmap")
        spec.output("points")
        
    def compute(self, op_input, op_output, context):
        image = op_input.receive("image")
        point = op_input.receive("point")
        
        self.diff_d_to_add[self.counter, :, :] = image
        self.points_to_add[:, self.counter] = point
        
        if self.counter < (self.batchsize - 1):    
            self.counter += 1
        else:
            print(f"{self.points_to_add[:, 0] = } <<<<<<<<<<<<<<<<<<<<<<<<<<<")
            op_output.emit(self.diff_d_to_add, "detmap")
            op_output.emit(self.points_to_add, "points")
            self.counter = 0
            # self.logger.info("Emitting data batch")
            

class ReconOp(Operator):
    def __init__(self, *args, param=None, postprocessing_flag=False, **kwargs):
        self.logger = logging.getLogger("ReconOp")
        logging.basicConfig(level=logging.INFO)
        super().__init__(*args, **kwargs)
        self.param = param
        self.recon = create_recon_object(param)
        
    def setup(self, spec):
        spec.input("detmap")
        spec.input("points")
        spec.output("result")
    
    def compute(self, op_input, op_output, context):
        diff_d_to_add = op_input.receive("detmap")
        points_to_add = op_input.receive("points")
        
        if self.recon.prb is None:
            deal_with_init_prb(self.recon, self.param, diff_d_to_add) 
        
        if not self.recon.is_setup:
            self.recon.recon_ptycho_init()

        self.recon.update_arrays(diff_d_to_add, points_to_add * -1)
        
        # self.logger.info("Reconstruction started")
        
        self.recon.recon_ptycho_run()
        # self.recon.quick_fig_save_for_test()
        output = self.recon.fetch_obj_ave()
        # self.logger.info("Reconstruction finished")
        
        op_output.emit(output, "result")
        # op_output.emit(1, "result")
    
    # def stop(self, *args, **kwargs):
    #     self.recon.finalize()
    #     super().stop(*args, **kwargs)


class BatchedResultStackerOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("BatchedResultStackerOp")
        logging.basicConfig(level=logging.INFO)
        self.batched_result = []
        
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        
    def compute(self, op_input, op_output, context):
        obj = op_input.receive("in")
        self.batched_result.append(obj)
        bla = np.array(self.batched_result)
        # self.logger.info(f"Emitting obj_data with shape={bla.shape}")
        out = np.nanmean(bla, axis=0)
        op_output.emit(out, "out")

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


class EigerPtychoAppBase(Application):
    def __init__(self, *args,
                 eiger_ip:str=None,
                 eiger_port:str=None,
                 msg_format:str=None,
                 simulate_position_data_stream:bool=None,
                 position_data_path:str=None,
                 recon_param=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eiger_ip = eiger_ip
        self.eiger_port = eiger_port
        self.msg_format = msg_format
        self.simulate_position_data_stream = simulate_position_data_stream
        self.position_data_path = position_data_path
        self.recon_param = recon_param
    
    def compose(self):
        eiger_zmq_rx = EigerZmqRxOp(self,
                                    eiger_ip=self.eiger_ip,
                                    eiger_port=self.eiger_port,
                                    msg_format=self.msg_format,
                                    simulate_position_data_stream=self.simulate_position_data_stream,
                                    name="eiger_zmq_rx")
        pos_rx = PositionRxOp(self,
                              simulate_position_data_stream=self.simulate_position_data_stream,
                              name="pos_rx")
        pos_proc_op = PositionPreprocessOp(self,
                                        name="pos_proc_op")
        preproc_op = ImagePreprocessOp(self, name="preproc_op")
        gather = GatherOp(self, name="gather")
        recon = ReconOp(self, param=self.recon_param,
                        postprocessing_flag=False,
                        name="recon")
        batch_stacker = BatchedResultStackerOp(self, name="batch_stacker")
        self.add_flow(eiger_zmq_rx, preproc_op, {("image", "image")})
        self.add_flow(preproc_op, gather, {("diff_amp", "image")})
        self.add_flow(pos_rx, pos_proc_op, {("point", "point")})
        self.add_flow(pos_proc_op, gather, {("point_rel", "point")})
        self.add_flow(gather, recon, {("detmap", "detmap"), ("points", "points")})
        self.add_flow(recon, batch_stacker)
        
        if self.simulate_position_data_stream:
            pos_sim_tx = PositionSimTxOp(self,
                                         position_data_path=self.position_data_path,
                                         name="pos_sim_tx")
            self.add_flow(eiger_zmq_rx, pos_sim_tx, {("count", "index")})
            self.add_flow(pos_sim_tx, pos_rx)
        
        self._eiger_zmq_rx_pointer = eiger_zmq_rx
        self._pos_rx_pointer = pos_rx
        self._graph_head_pointer = batch_stacker

class EigerPtychoApp(EigerPtychoAppBase):
    def compose(self):
        super().compose()
        sink = sink_func(self, name="sink")
        self.add_flow(self._graph_head_pointer, sink)


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

    app = EigerPtychoApp(
        eiger_ip=eiger_ip,
        eiger_port=eiger_port,
        msg_format=msg_format,
        simulate_position_data_stream=simulate_position_data_stream,
        position_data_path=position_data_path,
        recon_param=recon_param)
    
    # scheduler = EventBasedScheduler(
    #             app,
    #             worker_thread_number=16,
    #             stop_on_deadlock=True,
    #             stop_on_deadlock_timeout=500,
    #             name="event_based_scheduler",
    #         )
    scheduler = MultiThreadScheduler(
                app,
                worker_thread_number=8,
                check_recession_period_ms=0.5,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)
    
    app.run()
    
    
