import logging
from time import sleep

import numpy as np
import cupy as cp


from ..ptycho.utils import parse_config
from ..ptycho.recon_ptycho_gui import recon_gui

# from nsls2ptycho.core.ptycho.recon_ptycho_gui import create_recon_object, deal_with_init_prb
# from nsls2ptycho.core.ptycho.utils import parse_config
# from nsls2ptycho.core.ptycho_param import Param

from holoscan.core import Application, Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op

from .datasource import parse_args, EigerZmqRxOp, PositionRxOp
from .preprocess import ImageBatchOp, ImagePreprocessorOp, PointProcessorOp, ImageSendOp


class InitRecon(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.flag = True
    def setup(self,spec):
        spec.output("init")
    def compute(self,op_input,op_output,context):
        if self.flag:
            print("FIRE UP!!")
            op_output.emit(None,"init")
            self.flag = False
        else:
            return

class PtychoCtrl(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.pos_ready_num = 0
        self.frame_ready_num = 0

    def setup(self,spec):
        spec.input("ctrl_input").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
        spec.output("ready_num")

    def compute(self,op_input,op_output,context):
        data = op_input.receive("ctrl_input")
        if data:
            if data[0] == "pos":
                # print(f"Recv pos {data[1]}")
                self.pos_ready_num = data[1]
            
            if data[0] == "frame":
                # print(f"Recv frame {data[1]}")
                self.frame_ready_num = data[1]
        else:
            print(f"Recv pos {self.pos_ready_num} frame {self.frame_ready_num}")
            op_output.emit(np.minimum(self.pos_ready_num,self.frame_ready_num),"ready_num")


class PtychoRecon(Operator):
    def __init__(self, *args, param=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.recon, rank = recon_gui(param)
        self.recon.setup()

        self.num_points_min = 200
        self.it = 0
        self.it_last_update = np.inf



    def setup(self,spec):
        #spec.input("pos_ready_num",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.input("ready_num")
        spec.output("ctrl")
        spec.output("output").condition(ConditionType.NONE)

    def compute(self,op_input,op_output,context):

        ready_num = op_input.receive("ready_num")

        self.recon.num_points_recon = int(ready_num)

        if self.recon.num_points_recon > self.num_points_min:
            # Maybe not?
            self.recon.live_update_plan_last()

            self.recon.one_iter(self.it)
            self.it += 1

            if self.it_last_update == np.inf and self.recon.num_points_recon == self.recon.num_points:
                self.it_last_update = self.it
        else:
            sleep(2)


            # #save
            # if self.recon.num_points_recon >= 2500:
            #     print('saving..')
            #     np.save('diff_d.npy',self.recon.diff_d.get())
            #     np.save('point_info_d.npy',self.recon.point_info_d.get())
        
        if self.it - self.it_last_update >= 10:
            self.num_points_min = np.inf
            op_output.emit(self.recon.obj,"output")
        
        op_output.emit(None,"ctrl")

@create_op(inputs="output")
def SaveResult(output):
    print("Done!")
    np.save('/eiger_dir/live_test/output.npy',output)
    return
    

class PtychoApp(Application):
    def config_ops(self,param):

        nx_prb = self.pty.recon.nx_prb
        ny_prb = self.pty.recon.ny_prb
        nz = 2500

        batchsize = 100
        min_points = 200

        self.eiger_zmq_rx.roi = np.array([[644, 900], [525, 781]])

        self.image_batch.batchsize = batchsize
        self.image_batch.nx_prb = nx_prb
        self.image_batch.ny_prb = ny_prb
        self.image_batch.images_to_add = np.zeros((batchsize, 256, 256), dtype = np.uint32)
        self.image_batch.indices_to_add = np.zeros(batchsize, dtype=np.int32)

        self.image_proc.detmap_threshold = 0
        self.image_proc.badpixels = np.array([])

        self.image_send.diff_d_target = self.pty.recon.diff_d

        self.point_proc.point_info = np.zeros((nz,4),dtype = np.int32)
        self.point_proc.point_info_target = self.pty.recon.point_info_d

        self.point_proc.min_points = min_points
        self.point_proc.x_direction = self.pty.recon.x_direction
        self.point_proc.y_direction = self.pty.recon.y_direction
        self.point_proc.x_range_um = self.pty.recon.x_range_um
        self.point_proc.y_range_um = self.pty.recon.y_range_um
        self.point_proc.x_pixel_m = self.pty.recon.x_pixel_m
        self.point_proc.y_pixel_m = self.pty.recon.y_pixel_m
        self.point_proc.nx_prb = nx_prb
        self.point_proc.ny_prb = ny_prb
        self.point_proc.obj_pad = self.pty.recon.obj_pad

        self.pty.num_points_min = min_points



    def compose(self):

        param = parse_config('/eiger_dir/ptycho_holo/ptycho_config.txt')
        param.live_recon_flag = True

        self.eiger_zmq_rx = EigerZmqRxOp(self,"tcp://10.66.19.45:5559")

        self.pos_rx = PositionRxOp(self,endpoint = "tcp://10.66.19.45:6666", ch1 = "/INENC2.VAL.Value", ch2 = "/INENC3.VAL.Value", upsample_factor=10)

        self.image_batch = ImageBatchOp(self)
        self.image_proc = ImagePreprocessorOp(self)
        self.image_send = ImageSendOp(self)
        self.point_proc = PointProcessorOp(self)

        self.init_recon = InitRecon(self)
        self.pty = PtychoRecon(self,param=param,name='pty')
        self.pty_ctrl = PtychoCtrl(self)

        # Temp
        self.o = SaveResult(self,name='out')

        self.config_ops(param)


        self.add_flow(self.eiger_zmq_rx,self.image_batch,{("image","image"),("image_index","image_index")})
        self.add_flow(self.image_batch,self.image_proc,{("image_batch","image_batch"),("image_indices","image_indices_in")})
        self.add_flow(self.image_proc,self.image_send,{("diff_amp","diff_amp"),("image_indices","image_indices")})
        
        self.add_flow(self.pos_rx,self.point_proc,{("pointRx_out","pointOp_in")})
        self.add_flow(self.image_send,self.point_proc,{("image_indices_out","pointOp_in")})
    
        self.add_flow(self.init_recon,self.pty_ctrl,{("init","ctrl_input")})
        self.add_flow(self.image_send,self.pty_ctrl,{("frame_ready_num","ctrl_input")})
        self.add_flow(self.point_proc,self.pty_ctrl,{("pos_ready_num","ctrl_input")})
        self.add_flow(self.pty_ctrl,self.pty,{("ready_num","ready_num")})
        self.add_flow(self.pty,self.pty_ctrl,{("ctrl","ctrl_input")})



        self.add_flow(self.pty,self.o,{("output","output")})


def main():
    #config = parse_args()

    app = PtychoApp()
    #app.config()
    
    # scheduler = EventBasedScheduler(
    #             app,
    #             worker_thread_number=16,
    #             stop_on_deadlock=True,
    #             stop_on_deadlock_timeout=500,
    #             name="event_based_scheduler",
    #         )
    # app.scheduler(scheduler)
    
    scheduler = MultiThreadScheduler(
                app,
                worker_thread_number=20,
                check_recession_period_ms=0.001,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)
    
    app.run()
    
    
