import logging
import socket
import zmq
from argparse import ArgumentParser
import sys
import numpy as np
import numpy.typing as npt
import cupy as cp
import json
import cbor2
import pprint
import traceback
import h5py
import time

import copy
from dectris.compression import decompress

from holoscan.core import Application, Operator, OperatorSpec, Tracker, ConditionType, IOSpec
from holoscan.decorator import create_op
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.conditions import PeriodicCondition


def std_err_print(msg):
    sys.stderr.write(msg+"\n")


supported_encodings = {"bs32-lz4<": "bslz4", "lz4<": "lz4", "bs16-lz4<": "bslz4"}
supported_types = {"uint32": "uint32", "uint16": "uint16"}
def decode_json_message(data_msg, encoding_msg) -> tuple[str, npt.NDArray]:
    # std_err_print("DECODING THE MESSAGE")
    # There should be more robust way to detect this frame
    if "htype" in encoding_msg and encoding_msg["htype"] == "dimage_d-1.0":
        data_encoding = encoding_msg.get("encoding", None)
        data_shape = encoding_msg.get("shape", None)
        data_type = encoding_msg.get("type", None)

        data_encoding_str = supported_encodings.get(data_encoding, None)
        if not data_encoding_str:
            raise RuntimeError(f"Encoding {data_encoding!r} is not supported")

        data_type_str = supported_types.get(data_type, None)
        if not data_type_str:
            raise RuntimeError(f"Encoding {data_type!r} is not supported")

        elem_type = getattr(np, data_type_str)
        elem_size = elem_type(0).nbytes
        # std_err_print(f"data_msg: {data_msg}")
        decompressed = decompress(data_msg, data_encoding_str, elem_size=elem_size)
        image = np.frombuffer(decompressed, dtype=elem_type)
        image = image.reshape(data_shape[1], data_shape[0])
        msg_type = "image"
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
        image_id = msg['image_id']
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
        msg_type = msg["type"]
        image, image_id = None, None
    return msg_type, image, image_id


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

class EigerZmqRxOp(Operator):
    def __init__(self, fragment, endpoint = "" , msg_format = "json", receive_timeout_ms = 100,
                 simulate_position_data_stream = False, *args, **kwargs):
        
        
        self.endpoint = endpoint
        self.msg_format = msg_format
        # self.receive_times = []
        # self.roi = None
        self.simulate_position_data_stream = simulate_position_data_stream

        self.index = 0
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

        self.frame_id_last = -1

    def setup(self, spec: OperatorSpec):
        spec.input("flush").condition(ConditionType.NONE)
        spec.output("image").condition(ConditionType.NONE)
        spec.output("image_index").condition(ConditionType.NONE)
        # spec.output("image_index_encoding").condition(ConditionType.NONE)
        # if self.simulate_position_data_stream:
        #     spec.output("image_index").condition(ConditionType.NONE)
    
    def compute(self, op_input, op_output, context):
        # self.logger.info("Waiting for message")
        try:
            # Try to receive with timeout
            # msg = self.socket.recv()
            # self.logger.info(f"Received message: {msg}")
            
            if self.msg_format == "json":
                while True:
                    msg = self.socket.recv()
                    try: # skip messages that are not json
                        msg = json.loads(msg.decode())
                    except:
                        continue
                    if "frame" in msg:
                        break
                frame_id = msg["frame"]
                self.frame_id_last = frame_id
                # encoding info
                encoding_msg = self.socket.recv()
                encoding_msg = json.loads(encoding_msg.decode())
                data_msg = self.socket.recv()
                msg_type = "image"
                # _, image_data = decode_json_message(data_msg, encoding_msg)
                # self.receive_times.append(time.time())
                output = (copy.deepcopy(data_msg), copy.deepcopy(frame_id), copy.deepcopy(encoding_msg))
                op_output.emit(output, "image_index_encoding")
                
                # if len(self.receive_times) == 2000:
                #     _receive_times = np.array(self.receive_times)
                #     times_between_frames = np.diff(_receive_times)
                #     std_err_print(f"mean time between frames: {np.mean(times_between_frames)}")
                #     std_err_print(f"median time between frames: {np.median(times_between_frames)}")
                #     std_err_print(f"std time between frames: {np.std(times_between_frames)}")
                #     std_err_print(f"min time between frames: {np.min(times_between_frames)}")
                #     std_err_print(f"max time between frames: {np.max(times_between_frames)}")
                
                
                return

                    

                # std_err_print(f"time between image rx: {time.time() - self.receive_timeout_ms}")
                # image_data = image_data[self.roi[0, 0]:self.roi[0, 1],
                #                         self.roi[1, 0]:self.roi[1, 1]]
            elif self.msg_format == "cbor":
                msg = self.socket.recv()
                msg_type, image_data, image_id = decode_cbor_message(msg)

            if msg_type == "image":
                # output = {"image": image_data, "frame_id": image_id}
                op_output.emit(image_data, "image")
                op_output.emit(image_id, "image_index")
                # if self.simulate_position_data_stream:
                    # op_output.emit(image_id, "image_index")
                self.index += 1
            else: # probably should have a better handling of start/end messages
                self.index = 0

            # except Exception as ex:
            #     result = "ERROR: Failed to process message: {ex}"
            #     std_err_print(f"{pprint.pformat(result)}")
            #     std_err_print(traceback.format_exc())
                
        except zmq.error.Again:
            # Timeout occurred
            self.logger.debug("No message received within timeout period")
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            
    def __del__(self):
        """Cleanup socket on deletion"""
        if hasattr(self, 'socket'):
            self.socket.close()


class EigerDecompressOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("EigerDecompressOp")
        logging.basicConfig(level=logging.INFO)
        
    def setup(self, spec: OperatorSpec):
        spec.input("image_index_encoding").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=512, policy = IOSpec.QueuePolicy.POP)
        
        spec.output("decompressed_image")#.condition(ConditionType.NONE)
        spec.output("image_index")#.condition(ConditionType.NONE)
        
    def compute(self, op_input, op_output, context):
        compressed_image, image_index, encoding_msg = op_input.receive("image_index_encoding")
        _, decompressed_image = decode_json_message(compressed_image, encoding_msg)
        op_output.emit(decompressed_image, "decompressed_image")
        op_output.emit(image_index, "image_index")


class PositionSimTxOp(Operator):
    def __init__(self, *args, h5_file:str=None, upsample_factor:int=None, point_batch:int=200, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PositionTxOp")
        logging.basicConfig(level=logging.INFO)
        self.image_index = None
        self.current_index = 0
        self.frame_count = 0
        self.upsample_factor = upsample_factor
        self.point_batch = point_batch
        
        self.h5_file = h5_file
        with h5py.File(self.h5_file, 'r') as f:
            self.points = f["points"][()]
        self.logger.info(f"points: {self.points.shape}")
        
    def setup(self, spec: OperatorSpec):
        spec.input("image_index").condition(ConditionType.NONE).connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=1, policy = IOSpec.QueuePolicy.POP)
        spec.output("msg")
        
        
    def compute(self, op_input, op_output, context):
        image_index = op_input.receive("image_index")
        if self.image_index is None:
            self.image_index = image_index
        
        if self.image_index is not None:
            idx1 = self.current_index
            idx2 = np.min([self.current_index+self.point_batch, self.points.shape[1]])
            if idx1 == idx2:
                return
            # self.logger.info(f"{self.points[0, idx1:idx2][:10]=}")
            x_to_send = np.tile(self.points[0, idx1:idx2], (self.upsample_factor, 1)).T.ravel()
            y_to_send = np.tile(self.points[1, idx1:idx2], (self.upsample_factor, 1)).T.ravel()

            self.current_index = idx2

            msg = {
                "msg_type": "data",
                "frame_number": self.frame_count,
                "datasets": {
                    "/INENC2.VAL.Value": {"data": x_to_send.tolist()},
                    "/INENC3.VAL.Value": {"data": y_to_send.tolist()}
                }
            }
            # self.logger.info(f"emitting msg: {msg['msg_type']}")
            op_output.emit(msg, "msg")
            self.frame_count += 1


class PositionRxOp(Operator):
    def __init__(self, *args,
                endpoint:str=None,
                simulate_stream:bool=False,
                receive_timeout_ms:int=100,
                ch1:str=None,
                ch2:str=None,
                upsample_factor:int=None,
                **kwargs):
        
        self.logger = logging.getLogger("PositionRxOp")
        logging.basicConfig(level=logging.INFO)

        self.data_x_str = ch1
        self.data_y_str = ch2
        self.upsample_factor = upsample_factor
        self.endpoint = endpoint
        self.simulate_stream = simulate_stream
        if not simulate_stream:
            context = zmq.Context()
            socket = context.socket(zmq.SUB)
            socket.connect(self.endpoint)
            socket.setsockopt_string(zmq.SUBSCRIBE, "")
            # Set receive timeout
            socket.setsockopt(zmq.RCVTIMEO, receive_timeout_ms)
            self.socket = socket
        super().__init__(*args, **kwargs)
        
    def flush(self,param):
        self.data_x_str = param[0]
        self.data_y_str = param[1]

    def setup(self, spec: OperatorSpec):
        spec.input("flush",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.output("pointRx_out")
        if self.simulate_stream:
            spec.input("pointRx_in")
        
    def compute(self, op_input, op_output, context):
        param = op_input.receive('flush')
        if param:
            self.flush(param)
        if self.simulate_stream:
            pointRx_in = op_input.receive("pointRx_in")
            msg = pointRx_in
        else:
            try:
                msg = self.socket.recv_json()
                
            except zmq.error.Again:
            # Timeout occurred
                self.logger.debug("No message received within timeout period")
                return
            except Exception as e:
                self.logger.error(f"Error receiving message: {e}")
                return
        
        if msg["msg_type"] == "data":
            frame_number = msg["frame_number"]
            
            x = msg["datasets"][self.data_x_str]["data"]
            y = msg["datasets"][self.data_y_str]["data"]
            # self.logger.info(f"emitting x: {x.shape}, y: {y.shape}")
            op_output.emit((frame_number,np.array([x, y])), "pointRx_out")
            # self.logger.info(f"emitting x: {x[:20]}, y: {y[:20]}")
            # op_output.emit({"frame_number": frame_number,
            #                 "xy": np.array([x, y])},
            #                 "pointRx_out")

class SinkOp(Operator):
    def __init__(self, fragment, *args, silent=False, **kwargs):
        super().__init__(fragment, *args, **kwargs) 
        self.logger = logging.getLogger(kwargs.get("name", "SinkOp"))
        self._first_frame_time = None
        self._total_frame_count = 0
        self.silent = silent

    def setup(self, spec: OperatorSpec):
        spec.param("receivers", kind="receivers")
        

    def compute(self, op_input, op_output, context):
        receivers = op_input.receive("receivers")
        if self._first_frame_time is None:
            self._first_frame_time = time.time()
        self._total_frame_count += 1
        if self._total_frame_count % 100 == 0:
            elapsed = time.time() - self._first_frame_time
            self.logger.info(f"{self._total_frame_count} received in {elapsed:.1f}s. speed: {self._total_frame_count/elapsed:.1f} Hz")
        # print(f"msgs received: {len(receivers)}")
        if not self.silent:
            for i, receiver in enumerate(receivers):
                if isinstance(receiver, np.ndarray):
                    self.logger.info(f"msg {i}: tensor with shape {receiver.shape}")
                else:
                    for key, tensor in receiver.items():
                        if isinstance(tensor, np.ndarray):
                            self.logger.info(f"msg {i}: tensor {key} with shape {tensor.shape}")
                        else:
                            self.logger.info(f"msg {i}: tensor {key} and type {type(tensor)}")


class DataSourceApp(Application):
    def __init__(self, *args,
                 simulate_position_data_stream=False,
                 position_data_path=None,
                 **kwargs):
        super().__init__(*args,**kwargs)

        self.simulate_position_data_stream = simulate_position_data_stream
        self.position_data_path = position_data_path


    def compose(self):

        self.eiger_zmq_rx = EigerZmqRxOp(self,"tcp://0.0.0.0:5555", msg_format="cbor", name="eiger_zmq_rx",
                                         simulate_position_data_stream=self.simulate_position_data_stream)
        # self.eiger_decompress = EigerDecompressOp(self, name="eiger_decompress")
        
        if self.simulate_position_data_stream:
            print("Simulating position data stream")
            self.pos_tx = PositionSimTxOp(self,
                                          PeriodicCondition(self, recess_period=int(0.2*1e9)),
                                          upsample_factor=10,
                                          h5_file=self.position_data_path,
                                          name="pos_tx")
        self.pos_rx = PositionRxOp(self,
                                    simulate_stream=self.simulate_position_data_stream,
                                    endpoint = "tcp://0.0.0.0:5555",
                                    ch1 = "/INENC2.VAL.Value",
                                    ch2 = "/INENC3.VAL.Value",
                                    upsample_factor=10,
                                name="pos_rx")
        self.sink_img = SinkOp(self, silent=True, name="sink_img")
        self.sink_pos = SinkOp(self, name="sink_pos")

        # self.add_flow(self.eiger_zmq_rx,self.eiger_decompress,{("image_index_encoding","image_index_encoding")})
        # self.add_flow(self.eiger_decompress,self.sink,{("decompressed_image","decompressed_image")})

        self.add_flow(self.eiger_zmq_rx,self.sink_img,{("image","receivers"), ("image_index", "receivers")})
        self.add_flow(self.eiger_zmq_rx,self.pos_tx,{("image_index", "image_index")})
        self.add_flow(self.pos_tx,self.pos_rx,{("msg", "pointRx_in")})
        self.add_flow(self.pos_rx,self.sink_pos,{("pointRx_out","receivers")})

        

def main():
    app = DataSourceApp(
                simulate_position_data_stream=True,
                position_data_path="/test_data/scan_257331.h5")
    scheduler = MultiThreadScheduler(
                app,
                worker_thread_number=9,
                check_recession_period_ms=0.001,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                # strict_job_thread_pinning=True,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)
    
    app.run()        
# example of msg:
# {'msg_type': 'start', 'arm_time': '2025-02-28T18:44:22.905865051Z', 'start_time': '2025-02-28T18:44:22.905908989Z', 'hw_time_offset_ns': None}
# {'msg_type': 'data', 'frame_number': 0, 'datasets':
# {'/COUNTER1.OUT.Value': {'dtype': 'float64', 'size': 41, 'starting_sample_number': 0, 'data': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0]},
# '/FMC_IN.VAL1.Value': {'dtype': 'float64', 'size': 41, 'starting_sample_number': 0, 'data': [-0.00831604003356672, -0.007934570307256321, -0.00816345214304256, -0.007934570307256321, -0.00823974608830464, -0.00831604003356672, -0.00823974608830464, -0.00816345214304256, -0.008392333978828801, -0.00854492186935296, -0.00854492186935296, -0.00823974608830464, -0.00823974608830464, -0.00823974608830464, -0.008392333978828801, -0.008392333978828801, -0.00816345214304256, -0.00831604003356672, -0.00831604003356672, -0.00831604003356672, -0.008392333978828801, -0.00823974608830464, -0.00854492186935296, -0.00808715819778048, -0.00816345214304256, -0.008392333978828801, -0.00862121581461504, -0.007934570307256321, -0.008392333978828801, -0.00823974608830464, -0.00831604003356672, -0.00831604003356672, -0.00831604003356672, -0.008468627924090881, -0.00831604003356672, -0.0080108642525184, -0.00808715819778048, -0.008468627924090881, -0.008468627924090881, -0.00808715819778048, -0.007858276361994241]},
# '/PCAP.TS_TRIG.Value': {'dtype': 'float64', 'size': 41, 'starting_sample_number': 0, 'data': [2.4000000000000003e-08, 0.010000024000000001, 0.020000024, 0.030000024, 0.040000024, 0.050000024000000004, 0.060000024000000006, 0.07000002400000001, 0.080000024, 0.09000002400000001, 0.100000024, 0.110000024, 0.12000002400000001, 0.13000002400000002, 0.140000024, 0.150000024, 0.16000002400000002, 0.170000024, 0.180000024, 0.19000002400000002, 0.20000002400000003, 0.210000024, 0.22000002400000002, 0.23000002400000003, 0.240000024, 0.25000002400000004, 0.260000024, 0.270000024, 0.280000024, 0.290000024, 0.30000002400000003, 0.31000002400000004, 0.320000024, 0.330000024, 0.340000024, 0.350000024, 0.36000002400000003, 0.37000002400000004, 0.38000002400000005, 0.390000024, 0.400000024]}}}
# ...
# {'msg_type': 'stop', 'emitted_frames': 4}


            # data = np.array([0, 0]) # placeholder - this should be changed to something that will actually receive the data
        # op_output.emit(data, "point")
        # op_output.emit(index, "point_index")



