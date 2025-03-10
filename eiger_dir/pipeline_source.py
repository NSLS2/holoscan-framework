import logging
import socket
import zmq
from argparse import ArgumentParser

import numpy as np
import numpy.typing as npt
import cupy as cp
import json
import cbor2
import pprint
import traceback
import h5py

from dectris.compression import decompress

from holoscan.core import Application, Operator, OperatorSpec, Tracker
from holoscan.decorator import create_op
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler

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
                 receive_timeout_ms:int=1000,  # 1 second timeout
                 **kwargs):
        
        self.endpoint = f"tcp://{eiger_ip}:{eiger_port}"
        self.msg_format = msg_format
        self.index = 0
        self.simulate_position_data_stream = simulate_position_data_stream
        self.receive_timeout_ms = receive_timeout_ms
        context = zmq.Context()
        
        self.socket = context.socket(zmq.PULL)
        # Set receive timeout
        self.socket.setsockopt(zmq.RCVTIMEO, receive_timeout_ms)

        try:
            self.socket.connect(self.endpoint)
        except socket.error:
            self.logger.error("Failed to create socket")
        
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("EigerZmqRxOp")
        logging.basicConfig(level=logging.INFO)

    def setup(self, spec: OperatorSpec):
        spec.output("image")
        spec.output("image_index")

    def compute(self, op_input, op_output, context):
        # self.logger.info("Waiting for message")
        try:
            # Try to receive with timeout
            msg = self.socket.recv()
            # self.logger.info(f"Received message: {msg}")
            
            try:
                if self.msg_format == "json":
                    msg_type, image_data = decode_json_message(msg)
                elif self.msg_format == "cbor":
                    msg_type, image_data = decode_cbor_message(msg)

                if msg_type == "image":
                    op_output.emit(image_data, "image")
                    op_output.emit(self.index, "image_index")
                    self.index += 1
                else: # probably should have a better handling of start/end messages
                    self.index = 0

            except Exception as ex:
                result = "ERROR: Failed to process message: {ex}"
                print(f"{pprint.pformat(result)}")
                print(traceback.format_exc())
                
        except zmq.error.Again:
            # Timeout occurred
            self.logger.debug("No message received within timeout period")
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            
    def __del__(self):
        """Cleanup socket on deletion"""
        if hasattr(self, 'socket'):
            self.socket.close()


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
            # self.N = self.points.shape[0]
            # self.index = 0

    def setup(self, spec: OperatorSpec):
        spec.input("image_index")
        spec.output("point")
        spec.output("point_index")

    def compute(self, op_input, op_output, context):
        index = op_input.receive("image_index")
        point = self.points[index, :]
        op_output.emit(point, "point")
        op_output.emit(index, "point_index")
        # self.index += 1

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
            spec.input("index_input")
        spec.output("point")
        spec.output("point_index")
        
    def compute(self, op_input, op_output, context):
        if self.simulate_position_data_stream:
            data = np.asarray(op_input.receive("point_input"))
            index = op_input.receive("index_input")
        else:
            data = np.array([0, 0]) # placeholder - this should be changed to something that will actually receive the data
        op_output.emit(data, "point")
        op_output.emit(index, "point_index")

@create_op(inputs=("image", "point", "image_index", "point_index"))
def sink_func(image, point, image_index, point_index):
    print(f"SinkOp received image: {image.shape=}, {image_index=}")
    print(f"SinkOp received point: {point=}, {point_index=}")

class EigerRxBase(Application):
    def compose(self):
        simulate_position_data_stream = self.kwargs('eiger_zmq_rx')['simulate_position_data_stream']
        eiger_zmq_rx = EigerZmqRxOp(self, **self.kwargs('eiger_zmq_rx'))
        
        pos_rx_args = self.kwargs('pos_rx')
        pos_rx_args["simulate_position_data_stream"] = simulate_position_data_stream
        pos_rx = PositionRxOp(self, **pos_rx_args, name="pos_rx")
        
        if simulate_position_data_stream:
            pos_sim_tx = PositionSimTxOp(self, **self.kwargs('pos_sim_tx'))
            self.add_flow(eiger_zmq_rx, pos_sim_tx, {("image_index", "image_index")})
            self.add_flow(pos_sim_tx, pos_rx, {("point", "point_input"), ("point_index", "index_input")})

        return eiger_zmq_rx, pos_rx


class EigerRxApp(EigerRxBase):
    def compose(self):
        eiger_zmq_rx, pos_rx = super().compose()
        sink = sink_func(self, name="sink")
        
        self.add_flow(eiger_zmq_rx, sink, {("image", "image"), ("image_index", "image_index")})
        self.add_flow(pos_rx, sink, {("point", "point"), ("point_index", "point_index")})

def parse_args():
    parser = ArgumentParser(description="Eiger ingest example")
    parser.add_argument(
        "--config",
        type=str,
        default="none",
        help=(
            "Holoscan config file"
        ),
    )
    args = parser.parse_args()
    config = args.config
    if config == "none":
        config = "holoscan_config.yaml"
    return config
        
if __name__ == "__main__":
    config = parse_args()

    app = EigerRxApp()
    app.config(config)

    # scheduler = GreedyScheduler(
    #             app,
    #             max_duration_ms=37, # setting this to -1 will make the app run until all work is done; if positive number is given, the app will run for this amount of time (in ms)
    #             name="greedy_scheduler",
    #         )
    # app.scheduler(scheduler)

    # scheduler = EventBasedScheduler(
    #             app,
    #             worker_thread_number=16,
    #             max_duration_ms=5000,
    #             stop_on_deadlock=True,
    #             stop_on_deadlock_timeout=500,
    #             name="event_based_scheduler",
    #         )
    # app.scheduler(scheduler)
    scheduler = MultiThreadScheduler(
                app,
                worker_thread_number=4,
                # max_duration_ms=5000,
                check_recession_period_ms=0.001,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                strict_job_thread_pinning=True,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)
    app.run()
    
    # with Tracker(app, filename="logger.log", num_start_messages_to_skip=2, num_last_messages_to_discard=3) as tracker:
    #     app.run()
    #     tracker.print()
    
    

