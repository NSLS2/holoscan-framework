import sys
import time
import logging
import socket
import zmq

from time import sleep

import numpy as np
import cupy as cp
import json
import pprint
import traceback

from dectris.compression import decompress

from scipy import signal as cpu
from cupyx.scipy import signal as gpu

from holoscan.conditions import CountCondition
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.logger import LogLevel, set_log_level

eiger_ip = ...
eiger_port = "9999"

n_messages = 0

class UdpRxOp(Operator):
    sock_fd: socket.socket = None
    packet_size: int
    data: bytearray

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.logger = logging.getLogger("UdpRxOp")
        logging.basicConfig(level=logging.INFO)

        #self.packet_size = kwargs['packet_size'] * 8
        # Check if the packet size fits in a UDP packet.
        #assert self.packet_size <= 8000

    def initialize(self):
        Operator.initialize(self)
        self.logger.info("UDP RX Operator initialized")

        try:
            # self.sock_fd = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
            context = zmq.Context()
            # recieve work
            self.sock_fd = context.socket(zmq.PULL)
        except socket.error:
            self.logger.error("Failed to create socket")

        # self.sock_fd.bind(("127.0.0.1", 8080))
        self.sock_fd.connect(f"tcp://{eiger_ip}:{eiger_port}")

    def setup(self, spec: OperatorSpec):
        spec.output("wave")

    def compute(self, op_input, op_output, context):
        global n_messages
        data_encoding, data_shape, data_type, next_is_frame = None, None, None, False

        while True:
            #print(f"Waiting for data ...")
            #buffer = consumer_receiver.recv()
            buffer = self.sock_fd.recv()
            n_messages += 1
            #print(f"Message {n_messages} is received")

            try:
                if not next_is_frame:
                    buffer = json.loads(buffer.decode())

                    # There should be more robust way to detect this frame
                    if "htype" in buffer and buffer["htype"] == "dimage_d-1.0":
                        data_encoding = buffer.get("encoding", None)
                        data_shape = buffer.get("shape", None)
                        data_type = buffer.get("type", None)
                        next_is_frame = True

                    result = buffer

                else:
                    next_is_frame = False

                    try:
                        supported_encodings = {"bs32-lz4<": "bslz4", "lz4<": "lz4"}
                        data_encoding_str = supported_encodings.get(data_encoding, None)
                        if not data_encoding_str:
                            raise RuntimeError(f"Encoding {data_encoding!r} is not supported")

                        supported_types = {"uint32": "uint32"}
                        data_type_str = supported_types.get(data_type, None)
                        if not data_type_str:
                            raise RuntimeError(f"Encoding {data_type!r} is not supported")

                        elem_type = getattr(np, data_type_str)
                        elem_size = elem_type(0).nbytes
                        decompressed = decompress(buffer, data_encoding_str, elem_size=elem_size)

                        data = np.frombuffer(decompressed, dtype=elem_type)
                        #data = np.reshape(data, data_shape)
                        # The data should be properly shaped image of the respective type
                        result = f"Data frame is received. Image shape: {data.shape}"

                    except Exception as ex:
                        result = f"Failed to decode the received data frame. The frame size is {len(buffer)} bytes"
                        print(traceback.format_exc())

                    #wave = np.frombuffer(buffer, dtype=elem_type)
                    #op_output.emit(wave, "wave")
                    op_output.emit(data, "wave")
                    return
            except Exception as ex:
                result = "ERROR: Failed to process message: {ex}"
                print(traceback.format_exc())

            #print(f"{pprint.pformat(result)}")
            #print("=" * 80)


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

class SineUdpRx(Application):
    def compose(self):
        #batches = 8192
        batches = 1
        number_of_elements = 1030*1065
        resample_rate = 10
        use_cuda = '--cuda' in sys.argv

        if use_cuda:
            print("Using CUDA for resampling.")

        udp_rx_op = UdpRxOp(self, name="UdpRxOp", packet_size=number_of_elements)
        gather_op = GatherOp(self, name="GatherOp", batches=batches, input_shape=(number_of_elements,))
        resampler_op = ResamplerOp(self, name="ResamplerOp", rate=(number_of_elements // resample_rate), cuda=use_cuda)

        self.add_flow(udp_rx_op, gather_op)
        self.add_flow(gather_op, resampler_op)


def main():
    app = SineUdpRx()
    app.run()


if __name__ == "__main__":
    main()

