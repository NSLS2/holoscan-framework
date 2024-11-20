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

        data_type_str = self.supported_types.get(data_type, None)
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


class ZmqRxOp(Operator):
    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("ZmqRxOp")
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


    def compute(self, op_input, op_output, context):
        while True:
            msg = self.socket.recv()
            self.count += 1
            try:
                if self.msg_format == "json":
                    msg_type, image_data = decode_json_message(msg)
                elif self.msg_format == "cbor":
                    msg_type, image_data = decode_cbor_message(msg)

                if msg_type == "image":
                    self.logger.info(f"Successfully processed {self.count} frames")
                    op_output.emit(image_data, "image")
                else: # probably should have a better handling of start/end messages
                    self.logger.info("-" * 80)
                    self.logger.info("Image series start/end")
                    self.logger.info("-" * 80)
                    self.count = 0

            except Exception as ex:
                result = "ERROR: Failed to process message: {ex}"
                print(f"{pprint.pformat(result)}")
                print(traceback.format_exc())

@create_op
def sink_func(image):
    print(f"Sink received image ({image.shape=}")



class EigerPtychoApp(Application):
    def compose(self):
        zmq_rx_op = ZmqRxOp(self, name="zmq_rx_op")
        sink_op = sink_func(self, name="sink_op")

        self.add_flow(zmq_rx_op, sink_op)




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

    args = parser.parse_args()
    eiger_ip = args.eiger_ip
    eiger_port = args.eiger_port
    msg_format = args.message_format
    
    app = SineUdpRx()
    app.run()
