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


class GatherOp(Operator):

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("GatherOp")
        logging.basicConfig(level=logging.INFO)

        self.batches = kwargs['batches']
        self.input_shape = kwargs['input_shape']

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("Gather Operator initialized")

        self.iterator = 0
        print("input shape: ", *self.input_shape)
        self.data = np.zeros((self.batches, *self.input_shape), dtype=np.complex64)

    def setup(self, spec: OperatorSpec):
        spec.input("data_in")
        spec.output("data_out")

    def compute(self, op_input, op_output, context):
        data_in = op_input.receive("data_in")

        self.data[self.iterator, :] = data_in

        if self.iterator < self.batches - 1:
            self.iterator += 1
        else:
            op_output.emit(self.data, "data_out")
            self.iterator = 0

class ResamplerOp(Operator):

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("ResamplerOp")
        logging.basicConfig(level=logging.INFO)

        self.rate = kwargs['rate']
        self.cuda = kwargs['cuda']

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("Resampler Operator initialized")

    def setup(self, spec: OperatorSpec):
        spec.input("wave")

    def compute(self, op_input, op_output, context):
        wave = op_input.receive("wave")

        st = time.time()
        if self.cuda:
            print("using gpu for resampling")
            resampled_wave = gpu.resample(wave, self.rate, axis=1)
        else:
            print("using cpu for resampling")
            resampled_wave = cpu.resample(wave, self.rate, axis=1)
        et = time.time()

        print(wave.shape, resampled_wave.shape, f"Processing time: {(et - st) * 100} ms")

@create_op
def sink_func(image):
    print(f"Sink received image ({image.shape=}")



class SineUdpRx(Application):
    def compose(self):
        #batches = 8192
        batches = 1
        number_of_elements = 1030*1065
        resample_rate = 10
        use_cuda = '--cuda' in sys.argv

        if use_cuda:
            print("Using CUDA for resampling.")

        zmq_rx_op = ZmqRxOp(self, name="zmq_rx_op")
        sink_op = sink_func(self, name="sink_op")
        # gather_op = GatherOp(self, name="GatherOp", batches=batches, input_shape=(number_of_elements,))
        # resampler_op = ResamplerOp(self, name="ResamplerOp", rate=(number_of_elements // resample_rate), cuda=use_cuda)

        self.add_flow(zmq_rx_op, sink_op)
        # self.add_flow(gather_op, resampler_op)




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
