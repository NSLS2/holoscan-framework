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
                pandabox_ip:str=None,
                pandabox_port:str=None,
                receive_timeout_ms:int=1000,
                data_index_str:str=None,
                data_x_str:str=None,
                data_y_str:str=None,
                simulate_position_data_stream:bool=None,
                **kwargs):
        self.simulate_position_data_stream = simulate_position_data_stream
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PositionRxOp")
        logging.basicConfig(level=logging.INFO)
        self.data_index_str = data_index_str
        self.data_x_str = data_x_str
        self.data_y_str = data_y_str

        if not self.simulate_position_data_stream:
            self.endpoint = f"tcp://{pandabox_ip}:{pandabox_port}"
            context = zmq.Context()
            self.socket = context.socket(zmq.SUB)
            self.socket.connect(self.endpoint)
            self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
            # Set receive timeout
            self.socket.setsockopt(zmq.RCVTIMEO, receive_timeout_ms)

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
            msg = self.socket.recv_json()
            if msg["msg_type"] == "data":
                index = msg["datasets"][self.data_index_str]["data"]
                x = msg["datasets"][self.data_x_str]["data"]
                y = msg["datasets"][self.data_y_str]["data"]
                for index, x, y in zip(index, x, y):
                    op_output.emit(np.array([x, y]), "point")
                    op_output.emit(index, "point_index")
                    
# example of msg:
# {'msg_type': 'start', 'arm_time': '2025-02-28T18:44:22.905865051Z', 'start_time': '2025-02-28T18:44:22.905908989Z', 'hw_time_offset_ns': None}
# {'msg_type': 'data', 'frame_number': 0, 'datasets':
# {'/COUNTER1.OUT.Value': {'dtype': 'float64', 'size': 41, 'starting_sample_number': 0, 'data': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0]},
# '/FMC_IN.VAL1.Value': {'dtype': 'float64', 'size': 41, 'starting_sample_number': 0, 'data': [-0.00831604003356672, -0.007934570307256321, -0.00816345214304256, -0.007934570307256321, -0.00823974608830464, -0.00831604003356672, -0.00823974608830464, -0.00816345214304256, -0.008392333978828801, -0.00854492186935296, -0.00854492186935296, -0.00823974608830464, -0.00823974608830464, -0.00823974608830464, -0.008392333978828801, -0.008392333978828801, -0.00816345214304256, -0.00831604003356672, -0.00831604003356672, -0.00831604003356672, -0.008392333978828801, -0.00823974608830464, -0.00854492186935296, -0.00808715819778048, -0.00816345214304256, -0.008392333978828801, -0.00862121581461504, -0.007934570307256321, -0.008392333978828801, -0.00823974608830464, -0.00831604003356672, -0.00831604003356672, -0.00831604003356672, -0.008468627924090881, -0.00831604003356672, -0.0080108642525184, -0.00808715819778048, -0.008468627924090881, -0.008468627924090881, -0.00808715819778048, -0.007858276361994241]},
# '/PCAP.TS_TRIG.Value': {'dtype': 'float64', 'size': 41, 'starting_sample_number': 0, 'data': [2.4000000000000003e-08, 0.010000024000000001, 0.020000024, 0.030000024, 0.040000024, 0.050000024000000004, 0.060000024000000006, 0.07000002400000001, 0.080000024, 0.09000002400000001, 0.100000024, 0.110000024, 0.12000002400000001, 0.13000002400000002, 0.140000024, 0.150000024, 0.16000002400000002, 0.170000024, 0.180000024, 0.19000002400000002, 0.20000002400000003, 0.210000024, 0.22000002400000002, 0.23000002400000003, 0.240000024, 0.25000002400000004, 0.260000024, 0.270000024, 0.280000024, 0.290000024, 0.30000002400000003, 0.31000002400000004, 0.320000024, 0.330000024, 0.340000024, 0.350000024, 0.36000002400000003, 0.37000002400000004, 0.38000002400000005, 0.390000024, 0.400000024]}}}
# ...
# {'msg_type': 'stop', 'emitted_frames': 4}


            # data = np.array([0, 0]) # placeholder - this should be changed to something that will actually receive the data
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
    
    

