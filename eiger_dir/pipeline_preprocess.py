import logging

import numpy as np
import cupy as cp

from holoscan.core import Operator, OperatorSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op, Input

from pipeline_source import EigerRxBase, parse_source_args

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
        self.batchsize = 512
        
        self.diff_d_to_add = np.zeros((self.batchsize, 256, 256))
        self.points_to_add = np.zeros((2, self.batchsize))

    def setup(self, spec: OperatorSpec):
        spec.input("image")
        spec.input("point")
        spec.output("batch")
        
    def compute(self, op_input, op_output, context):
        image = op_input.receive("image")
        point = op_input.receive("point")
        
        self.diff_d_to_add[self.counter, :, :] = image
        self.points_to_add[:, self.counter] = point
        
        if self.counter < (self.batchsize - 1):    
            self.counter += 1
        else:
            out = {"images": self.diff_d_to_add,
                   "points": self.points_to_add}
            
            op_output.emit(out, "batch")
            self.counter = 0
            # self.logger.info("Emitting data batch")
            

@create_op(inputs=Input("batch", arg_map={"images": "images", "points": "points"}))
def sink_func(images, points):
    print(f"SinkOp received images: shape={images.shape}")
    print(f"SinkOp received points: shape={points.shape}")

class PreprocAppBase(EigerRxBase):
    def compose(self):
        eiger_zmq_rx, pos_rx = super().compose()
        
        pos_proc_op = PositionPreprocessOp(self,
                                        name="pos_proc_op")
        preproc_op = ImagePreprocessOp(self, name="preproc_op")
        gather = GatherOp(self, name="gather")
        
        self.add_flow(eiger_zmq_rx, preproc_op, {("image", "image")})
        self.add_flow(preproc_op, gather, {("diff_amp", "image")})

        self.add_flow(pos_rx, pos_proc_op, {("point", "point")})
        self.add_flow(pos_proc_op, gather, {("point_rel", "point")})
        
        return eiger_zmq_rx, pos_rx, gather
        

class PreprocApp(PreprocAppBase):
    def compose(self):
        _, _, gather = super().compose()
        sink = sink_func(self, name="sink")
        self.add_flow(gather, sink)


if __name__ == "__main__":
    eiger_ip, eiger_port, msg_format, simulate_position_data_stream, position_data_path = parse_source_args()
    
    app = PreprocApp(
        eiger_ip=eiger_ip,
        eiger_port=eiger_port,
        msg_format=msg_format,
        simulate_position_data_stream=simulate_position_data_stream,
        position_data_path=position_data_path)
    
    # # scheduler = EventBasedScheduler(
    # #             app,
    # #             worker_thread_number=16,
    # #             stop_on_deadlock=True,
    # #             stop_on_deadlock_timeout=500,
    # #             name="event_based_scheduler",
    # #         )
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
    
    
