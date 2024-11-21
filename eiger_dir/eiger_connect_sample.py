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

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op


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
            spec.output("trigger")


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
                        op_output.emit(1, "trigger")
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
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PositionSimTxOp")
        logging.basicConfig(level=logging.INFO)
        with h5py.File(position_data_path) as f:
            self.points = f["points"][()].T
            self.index = 0
            self.N = self.points.shape[0]

    def setup(self, spec: OperatorSpec):
        spec.input("trigger")
        spec.output("point")

    def compute(self, op_input, op_output, context):
        op_input.receive("trigger")
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





@create_op(inputs=("image", "point"))
def sink_func(image, point):
    print(f"SinkOp received image: shape={image.shape}, total_counts={np.sum(image)}")
    print(f"SinkOp received point coordinates: {point=}")



class EigerPtychoApp(Application):
    def compose(self):
        eiger_zmq_rx = EigerZmqRxOp(self, name="eiger_zmq_rx")
        pos_rx = PositionRxOp(self, name="pos_rx")
        sink = sink_func(self, name="sink")

        self.add_flow(eiger_zmq_rx, sink, {("image", "image")})
        self.add_flow(pos_rx, sink, {("point", "point")})

        if simulate_position_data_stream:
            pos_sim_tx = PositionSimTxOp(self, name="pos_sim_tx")
            self.add_flow(eiger_zmq_rx, pos_sim_tx, {("trigger", "trigger")})
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


    app = EigerPtychoApp()
    app.run()
