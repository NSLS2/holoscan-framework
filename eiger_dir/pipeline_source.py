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

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.decorator import create_op

# from holoscan.logger import LogLevel, set_log_level
# set_log_level(LogLevel.DEBUG)

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
                        pass
                        op_output.emit(self.count, "count") # emit count to trigger the position transmitter to emit corresponding point
                else: # probably should have a better handling of start/end messages
                    pass

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
            data = np.asarray(op_input.receive("point_input"))
        else:
            data = np.array([0, 0]) # placeholder - this should be changed to something that will actually receive the data
        # self.logger.info(f"Emitting point data {data}")
        op_output.emit(data, "point")

@create_op(inputs=("image", "point"))
def sink_func(image, point):
    print(f"SinkOp received image: shape={image.shape}, total_counts={np.sum(image)}")
    print(f"SinkOp received point coordinates: {point=}")

class EigerRxBase(Application):
    def __init__(self, *args,
                 eiger_ip:str=None,
                 eiger_port:str=None,
                 msg_format:str=None,
                 simulate_position_data_stream:bool=None,
                 position_data_path:str=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.eiger_ip = eiger_ip
        self.eiger_port = eiger_port
        self.msg_format = msg_format
        self.simulate_position_data_stream = simulate_position_data_stream
        self.position_data_path = position_data_path
            
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
        
        if self.simulate_position_data_stream:
            pos_sim_tx = PositionSimTxOp(self,
                                         position_data_path=self.position_data_path,
                                         name="pos_sim_tx")
            self.add_flow(eiger_zmq_rx, pos_sim_tx, {("count", "index")})
            self.add_flow(pos_sim_tx, pos_rx)
        
        return eiger_zmq_rx, pos_rx


class EigerRxApp(EigerRxBase):
    def compose(self):
        eiger_zmq_rx, pos_rx = super().compose()
        sink = sink_func(self, name="sink")
        
        self.add_flow(eiger_zmq_rx, sink, {("image", "image")})
        self.add_flow(pos_rx, sink, {("point", "point")})
        
        


def parse_source_args():
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
    return eiger_ip, eiger_port, msg_format, simulate_position_data_stream, position_data_path


if __name__ == "__main__":
    
    eiger_ip, eiger_port, msg_format, simulate_position_data_stream, position_data_path = parse_source_args()
    app = EigerRxApp(
        eiger_ip=eiger_ip,
        eiger_port=eiger_port,
        msg_format=msg_format,
        simulate_position_data_stream=simulate_position_data_stream,
        position_data_path=position_data_path,
        )
    
    app.run()
    
    

