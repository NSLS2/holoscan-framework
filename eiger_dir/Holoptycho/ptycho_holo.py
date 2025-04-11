import logging
from time import sleep

import sys
import os

import numpy as np
import cupy as cp

from hxntools.motor_info import motor_table


from ..ptycho.utils import parse_config
from ..ptycho.recon_ptycho_gui import recon_gui

# from nsls2ptycho.core.ptycho.recon_ptycho_gui import create_recon_object, deal_with_init_prb
# from nsls2ptycho.core.ptycho.utils import parse_config
# from nsls2ptycho.core.ptycho_param import Param

from holoscan.core import Application, Operator, OperatorSpec, ConditionType, IOSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op

from .datasource import parse_args, EigerZmqRxOp, PositionRxOp, EigerDecompressOp
from .preprocess import ImageBatchOp, ImagePreprocessorOp, PointProcessorOp, ImageSendOp
from .liverecon_utils import parse_scan_header

class InitRecon(Operator):
    def __init__(self, *args, param, scan_header_file, **kwargs):
        super().__init__(*args,**kwargs)
        self.scan_num = None
        self.scan_header_file = scan_header_file
        p = parse_scan_header(self.scan_header_file)
        if p:
            self.scan_num = p.scan_num

        self.roi_ptyx0 = param.batch_x0
        self.roi_ptyy0 = param.batch_y0
        self.nx = param.nx
        self.ny = param.ny

    def setup(self,spec):
        spec.output("flush_pos_rx").condition(ConditionType.NONE)
        spec.output("flush_image_batch").condition(ConditionType.NONE)
        spec.output("flush_image_send").condition(ConditionType.NONE)
        spec.output("flush_pos_proc").condition(ConditionType.NONE)
        spec.output("flush_pty").condition(ConditionType.NONE)

    def compute(self,op_input,op_output,context):
        p = parse_scan_header(self.scan_header_file)

        if p:
            if self.scan_num != p.scan_num:

                self.scan_num = p.scan_num
                # New scan
                op_output.emit((motor_table[p.x_motor][2],motor_table[p.y_motor][2]),'flush_pos_rx')
                op_output.emit([[p.det_roiy0 + self.roi_ptyy0, \
                                           p.det_roiy0 + self.roi_ptyy0 + self.ny],\
                                          [p.det_roix0 + self.roi_ptyx0, \
                                           p.det_roix0 + self.roi_ptyx0 + self.nx]],\
                                            'flush_image_batch')
                op_output.emit(True,'flush_image_send')
                op_output.emit((p.x_range,p.y_range,motor_table[p.x_motor][1],motor_table[p.y_motor][1],p.x_num*2),'flush_pos_proc')
                op_output.emit((p.x_range,p.y_range,p.x_num*2,p.nz),'flush_pty')
        sleep(0.05)

# class PtychoCtrl(Operator):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args,**kwargs)
#         self.pos_ready_num = 0
#         self.frame_ready_num = 0

#     def setup(self,spec):
#         spec.input("ctrl_input").connector(IOSpec.ConnectorType.DOUBLE_BUFFER, capacity=32)
#         spec.output("ready_num")

#     def compute(self,op_input,op_output,context):
#         data = op_input.receive("ctrl_input")
#         if data:
#             if data[0] == "pos":
#                 # print(f"Recv pos {data[1]}")
#                 self.pos_ready_num = data[1]
            
#             if data[0] == "frame":
#                 # print(f"Recv frame {data[1]}")
#                 self.frame_ready_num = data[1]
#         else:
#             print(f"Recv pos {self.pos_ready_num} frame {self.frame_ready_num}")
#             op_output.emit(np.minimum(self.pos_ready_num,self.frame_ready_num),"ready_num")


class PtychoRecon(Operator):
    def __init__(self, *args, param=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.recon, rank = recon_gui(param)
        self.recon.setup()

        self.num_points_min = 200
        self.it = 0
        self.it_last_update = np.inf
        self.it_ends_after = 10
        self.pos_ready_num = 0
        self.frame_ready_num = 0
        self.points_total = 0
    
    def flush(self,param):

        print('flush ptycho recon')
        self.it = 0
        self.it_last_update = np.inf
        self.pos_ready_num = 0
        self.frame_ready_num = 0
        self.recon.num_points_recon = 0

        self.recon.x_range_um = param[0]
        self.recon.y_range_um = param[1]

        self.num_points_min = param[2]
        self.points_total = param[3]

        nx_obj_new = int(self.recon.nx_prb + np.ceil(self.recon.x_range_um*1e-6/self.recon.x_pixel_m) + self.recon.obj_pad)
        ny_obj_new = int(self.recon.ny_prb + np.ceil(self.recon.y_range_um*1e-6/self.recon.y_pixel_m) + self.recon.obj_pad)

        if np.abs(self.recon.nx_obj - nx_obj_new) < self.recon.obj_pad and np.abs(self.recon.ny_obj - ny_obj_new) < self.recon.obj_pad:
            # Similar FOV, flush obj array without reinit
            self.recon.flush_obj()
        else:
            # Oops, new obj array is needed. Good luck..
            self.recon.new_obj()
            print('reload shared memory')

        # self.recon.init_mmap()

    def setup(self,spec):
        spec.input("flush",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.input("pos_ready_num",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        spec.input("frame_ready_num",policy=IOSpec.QueuePolicy.POP).condition(ConditionType.NONE)
        #spec.input("ready_num")
        spec.output("output").condition(ConditionType.NONE)

    def compute(self,op_input,op_output,context):

        flush_param = op_input.receive('flush')
        if flush_param:
            self.flush(flush_param)

        # ready_num = op_input.receive("ready_num")

        # self.recon.num_points_recon = int(ready_num)

        pos_ready_num = op_input.receive("pos_ready_num")
        

        if pos_ready_num:
            self.pos_ready_num = int(pos_ready_num)

        frame_ready_num = op_input.receive("frame_ready_num")

        if frame_ready_num:
            self.frame_ready_num = int(frame_ready_num)

        if self.it - self.it_last_update < self.it_ends_after and self.points_total>0:
            print(f"Recv pos {self.pos_ready_num} frame {self.frame_ready_num}")

        ready_num = np.minimum(self.pos_ready_num,self.frame_ready_num)

        if ready_num > self.recon.num_points_recon:
            self.recon.num_points_recon = ready_num
            if ready_num > self.points_total*0.8:
                self.it_last_update = self.it
        
        if self.recon.num_points_recon > self.num_points_min:
            # Maybe not?
            self.recon.live_update_plan_last()

            self.recon.one_iter(self.it)
            self.it += 1

        else:
            sleep(0.2)


            # #save
            # if self.recon.num_points_recon >= 2500:
            #     print('saving..')
            #     np.save('diff_d.npy',self.recon.diff_d.get())
            #     np.save('point_info_d.npy',self.recon.point_info_d.get())
        
        if self.it - self.it_last_update >= self.it_ends_after and self.num_points_min<np.inf:
            self.num_points_min = np.inf
            op_output.emit(self.recon.obj,"output")
        sys.stdout.flush()
        sys.stderr.flush()
        

@create_op(inputs="output")
def SaveResult(output):
    print('Done! Saving..')
    np.save('/eiger_dir/live_test/output.npy',output)
    return
    

class PtychoApp(Application):
    def __init__(self, *args, config_path=None, **kwargs):
        super().__init__(*args,**kwargs)

        self.config_path = config_path
    def config_ops(self,param):

        nx_prb = self.pty.recon.nx_prb
        ny_prb = self.pty.recon.ny_prb
        nz = self.pty.recon.num_points

        batchsize = 100
        min_points = 200

        self.image_batch.roi = np.array([[644, 900], [525, 781]])
        self.image_batch.batchsize = batchsize
        self.image_batch.nx_prb = nx_prb
        self.image_batch.ny_prb = ny_prb
        self.image_batch.images_to_add = np.zeros((batchsize, 256, 256), dtype = np.uint32)
        self.image_batch.indices_to_add = np.zeros(batchsize, dtype=np.int32)

        self.image_proc.detmap_threshold = 0
        self.image_proc.badpixels = np.array([])

        self.image_send.diff_d_target = self.pty.recon.diff_d
        self.image_send.max_points = nz

        self.point_proc.point_info = np.zeros((nz,4),dtype = np.int32)
        self.point_proc.point_info_target = self.pty.recon.point_info_d

        self.point_proc.min_points = min_points
        self.point_proc.max_points = nz
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

        param = parse_config(self.config_path)
        param.live_recon_flag = True

        self.eiger_zmq_rx = EigerZmqRxOp(self,"tcp://10.66.19.45:5559", name="eiger_zmq_rx")
        self.eiger_decompress = EigerDecompressOp(self, name="eiger_decompress")
        self.pos_rx = PositionRxOp(self,endpoint = "tcp://10.66.19.45:6666", ch1 = "/INENC2.VAL.Value", ch2 = "/INENC3.VAL.Value", upsample_factor=10,
                                   name="pos_rx")

        self.image_batch = ImageBatchOp(self, name="image_batch")
        self.image_proc = ImagePreprocessorOp(self, name="image_proc")
        self.image_send = ImageSendOp(self, name="image_send")
        self.point_proc = PointProcessorOp(self, name="point_proc")

        # self.init_recon = InitRecon(self)
        self.pty = PtychoRecon(self,param=param,name='pty')
        # self.pty_ctrl = PtychoCtrl(self)

        self.init = InitRecon(self,param=param,scan_header_file='/nsls2/data2/hxn/legacy/users/startup_parameters/scan_header.txt')

        # Temp
        self.o = SaveResult(self,name='out')

        self.config_ops(param)


        self.add_flow(self.eiger_zmq_rx,self.eiger_decompress,{("image_index_encoding","image_index_encoding")})
        self.add_flow(self.eiger_decompress,self.image_batch,{("decompressed_image","image"),("image_index","image_index")})
        self.add_flow(self.image_batch,self.image_proc,{("image_batch","image_batch"),("image_indices","image_indices_in")})
        self.add_flow(self.image_proc,self.image_send,{("diff_amp","diff_amp"),("image_indices","image_indices")})
        
        self.add_flow(self.pos_rx,self.point_proc,{("pointRx_out","pointOp_in")})
        self.add_flow(self.image_send,self.point_proc,{("image_indices_out","pointOp_in")})

        self.add_flow(self.image_send,self.pty,{("frame_ready_num","frame_ready_num")})
        self.add_flow(self.point_proc,self.pty,{("pos_ready_num","pos_ready_num")})

        self.add_flow(self.init,self.pos_rx,{("flush_pos_rx","flush")})
        self.add_flow(self.init,self.image_batch,{("flush_image_batch","flush")})
        self.add_flow(self.init,self.image_send,{("flush_image_send","flush")})
        self.add_flow(self.init,self.point_proc,{("flush_pos_proc","flush")})
        self.add_flow(self.init,self.pty,{("flush_pty","flush")})

        # pool1 = self.make_thread_pool("pool1", 1)
        # pool1.add(self.eiger_zmq_rx, True)
    
        # pool2 = self.make_thread_pool("pool2", 7)
        # pool2.add(self.pos_rx, True)
        # pool2.add(self.image_batch, True)
        # pool2.add(self.image_proc, True)
        # pool2.add(self.image_send, True)
        # pool2.add(self.point_proc, True)
        # pool2.add(self.pty, True)
        # pool2.add(self.o, True)
        

        # self.add_flow(self.init_recon,self.pty_ctrl,{("init","ctrl_input")})
        # self.add_flow(self.image_send,self.pty_ctrl,{("frame_ready_num","ctrl_input")})
        # self.add_flow(self.point_proc,self.pty_ctrl,{("pos_ready_num","ctrl_input")})
        # self.add_flow(self.pty_ctrl,self.pty,{("ready_num","ready_num")})
        # self.add_flow(self.pty,self.pty_ctrl,{("ctrl","ctrl_input")})



        self.add_flow(self.pty,self.o,{("output","output")})


def main():
    if len(sys.argv) == 1: # started from commmandline
        # raise NotImplementedError("No config file for Holoptycho")
        config_path = '/eiger_dir/ptycho_holo/ptycho_config.txt'
    elif len(sys.argv) == 2: # started from GUI
        config_path = sys.argv[1]
    #config = parse_args()

    app = PtychoApp(config_path=config_path)
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
                worker_thread_number=9,
                check_recession_period_ms=0.001,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                # strict_job_thread_pinning=True,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)
    
    app.run()
    
    
