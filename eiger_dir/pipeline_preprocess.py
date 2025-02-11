import logging

import numpy as np
import cupy as cp

from holoscan.core import Operator, OperatorSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op, Input

from pipeline_source import EigerRxBase, parse_source_args

class ImageBatchOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("ImageBatchOp")
        logging.basicConfig(level=logging.INFO)
        self.counter = 0
        self.batchsize = 512
        self.images_to_add = cp.zeros((self.batchsize, 257, 257))

    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.output("image_batch")
        
    def compute(self, op_input, op_output, context):
        image = op_input.receive("image")
        self.images_to_add[self.counter, :, :] = cp.array(image)
        
        if self.counter < (self.batchsize - 1):
            self.counter += 1
        else:
            op_output.emit(self.images_to_add, "image_batch")
            self.counter = 0

class PointBatchOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PointBatchOp")
        logging.basicConfig(level=logging.INFO)
        self.counter = 0
        self.batchsize = 512
        self.points_to_add = cp.zeros((2, self.batchsize))

    def setup(self, spec: OperatorSpec):
        spec.input("point")
        spec.output("point_batch")
        
    def compute(self, op_input, op_output, context):
        point = op_input.receive("point")
        self.points_to_add[:, self.counter] = cp.array(point)
        
        if self.counter < (self.batchsize - 1):
            self.counter += 1
        else:
            op_output.emit(self.points_to_add, "point_batch")
            self.counter = 0

class ImagePreprocessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("ImagePreprocessorOp")
        logging.basicConfig(level=logging.INFO)
        self.roi = np.array([[1, 257], [1,257]])
        self.detmap_threshold = 0
        self.badpixels = np.array([[207], [211]])
        
    def setup(self, spec: OperatorSpec):
        spec.input("image_batch")
        spec.output("diff_amp")
        
    def compute(self, op_input, op_output, context):
        images = op_input.receive("image_batch")
        processed_images = cp.asarray(images)
        
        for bd in self.badpixels.T:
            x = int(bd[0])
            y = int(bd[1])
            processed_images[:, x, y] = cp.median(processed_images[:, x-1:x+2, y-1:y+2], axis=(2, 1))
        
        processed_images = processed_images[:, self.roi[0,0]:self.roi[0,1], self.roi[1,0]:self.roi[1,1]]
        processed_images = cp.rot90(processed_images, axes=(2,1))
        processed_images = cp.fft.fftshift(processed_images, axes=(2,1))
        if self.detmap_threshold > 0:
            processed_images[processed_images<self.detmap_threshold] = 0
        diff_amp = cp.sqrt(processed_images)
        
        op_output.emit(diff_amp, "diff_amp")

class PointPreprocessorOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("PointPreprocessorOp")
        logging.basicConfig(level=logging.INFO)
        
    def setup(self, spec: OperatorSpec):
        spec.input("point_batch")
        spec.output("processed_points")
        
    def compute(self, op_input, op_output, context):
        points = op_input.receive("point_batch")
        # Add any position processing here if needed in the future
        op_output.emit(points, "processed_points")

class DataGatherOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("DataGatherOp")
        logging.basicConfig(level=logging.INFO)
        
    def setup(self, spec: OperatorSpec):
        spec.input("diff_amp")
        spec.input("points")
        spec.output("batch")
        
    def compute(self, op_input, op_output, context):
        diff_amp = op_input.receive("diff_amp")
        points = op_input.receive("points")
        # Here we should add filtering of images/points to deal with missing frames
        out = {"diff_amp": diff_amp, "points": points}
        op_output.emit(out, "batch")


@create_op(inputs=Input("batch", arg_map={"diff_amp": "images", "points": "points"}))
def sink_func(images, points):
    print(f"SinkOp received images: shape={images.shape}")
    print(f"SinkOp received points: shape={points.shape}")

class PreprocAppBase(EigerRxBase):
    def compose(self):
        eiger_zmq_rx, pos_rx = super().compose()
        
        # Create operators
        img_batch_op = ImageBatchOp(self, name="img_batch_op")
        point_batch_op = PointBatchOp(self, name="point_batch_op")
        img_proc_op = ImagePreprocessorOp(self, name="img_proc_op")
        point_proc_op = PointPreprocessorOp(self, name="point_proc_op")
        gather_op = DataGatherOp(self, name="gather_op")
        
        # Connect source operators to batch operators
        self.add_flow(eiger_zmq_rx, img_batch_op, {("image", "image")})
        self.add_flow(pos_rx, point_batch_op, {("point", "point")})
        
        # Connect batch operators to preprocessing operators
        self.add_flow(img_batch_op, img_proc_op, {("image_batch", "image_batch")})
        self.add_flow(point_batch_op, point_proc_op, {("point_batch", "point_batch")})
        
        self.add_flow(img_proc_op, gather_op, {("diff_amp", "diff_amp")})
        self.add_flow(point_proc_op, gather_op, {("processed_points", "points")})
        
        return eiger_zmq_rx, pos_rx, gather_op

class PreprocApp(PreprocAppBase):
    def compose(self):
        _, _, gather_op = super().compose()
        sink = sink_func(self, name="sink")
        self.add_flow(gather_op, sink)

if __name__ == "__main__":
    eiger_ip, eiger_port, msg_format, simulate_position_data_stream, position_data_path = parse_source_args()
    
    app = PreprocApp(
        eiger_ip=eiger_ip,
        eiger_port=eiger_port,
        msg_format=msg_format,
        simulate_position_data_stream=simulate_position_data_stream,
        position_data_path=position_data_path)
    
    # scheduler = EventBasedScheduler(
    #             app,
    #             worker_thread_number=16,
    #             stop_on_deadlock=True,
    #             stop_on_deadlock_timeout=500,
    #             name="event_based_scheduler",
    #         )
    # scheduler = MultiThreadScheduler(
    #             app,
    #             worker_thread_number=8,
    #             check_recession_period_ms=0.5,
    #             stop_on_deadlock=True,
    #             stop_on_deadlock_timeout=500,
    #             name="multithread_scheduler",
    #         )
    # app.scheduler(scheduler)
    
    app.run()
    
    
