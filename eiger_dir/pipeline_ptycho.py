
import logging
from time import sleep

import numpy as np
import cupy as cp


from nsls2ptycho.core.ptycho.recon_ptycho_gui import create_recon_object, deal_with_init_prb
from nsls2ptycho.core.ptycho.utils import parse_config
from nsls2ptycho.core.ptycho_param import Param

from holoscan.core import Application, Operator, OperatorSpec
from holoscan.schedulers import GreedyScheduler, MultiThreadScheduler, EventBasedScheduler
from holoscan.logger import LogLevel, set_log_level
from holoscan.decorator import create_op
# from holoscan.operators import HolovizOp

from pipeline_source import parse_source_args
from pipeline_preprocess import PreprocAppBase

import nvtx

class ReconOp(Operator):
    def __init__(self, *args, param=None, postprocessing_flag=False, **kwargs):
        self.logger = logging.getLogger("ReconOp")
        logging.basicConfig(level=logging.INFO)
        super().__init__(*args, **kwargs)
        self.param = param
        self.recon = create_recon_object(param)
        
    def setup(self, spec):
        spec.input("batch")
        spec.output("result")
    
    def compute(self, op_input, op_output, context):
        tensor = op_input.receive("batch")
        diff_d_to_add = cp.asarray(tensor["diff_amp"])
        points_to_add = cp.asarray(tensor["points"])
        
        if np.all(self.recon.prb == 0): # if probe is trivial it means that it was not fully initialized
            with nvtx.annotate("deal_with_init_prb", color="red"):
                # deal_with_init_prb(self.recon, self.param, diff_d_to_add)
                prb_init = deal_with_init_prb(self.recon, self.param, diff_d_to_add)
                self.recon.prb += cp.asnumpy(prb_init.astype(self.recon.complex_precision))
                # self.recon.prb_d[:, :] += cp.array(prb_init, dtype=self.recon.complex_precision, order='C')
                self.recon.prb_d[:, :] += prb_init.astype(self.recon.complex_precision)
        
        # if not self.recon.is_setup:
        #     with nvtx.annotate("self.recon.recon_ptycho_init()", color="yellow"):
        #         self.recon.recon_ptycho_init()
        
        with nvtx.annotate("self.recon.update_arrays", color="green"):
            self.recon.update_arrays(diff_d_to_add, points_to_add * -1)
        
        # self.logger.info("Reconstruction started")
        self.recon.recon_ptycho_run()
        # self.recon.quick_fig_save_for_test()
        output = self.recon.fetch_obj_ave()
        # self.logger.info("Reconstruction finished")
        
        op_output.emit(output, "result")


class BatchedResultStackerOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("BatchedResultStackerOp")
        logging.basicConfig(level=logging.INFO)
        self.batched_result = []
        
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        
    def compute(self, op_input, op_output, context):
        obj = op_input.receive("in")
        self.batched_result.append(obj)
        bla = np.array(self.batched_result)
        # self.logger.info(f"Emitting obj_data with shape={bla.shape}")
        out = np.nanmean(bla, axis=0)
        op_output.emit(out, "out")


@create_op(inputs="result")
def sink_func(result):
    print(f"SinkOp received the result from reconstruction {result=}")


class PtychoAppBase(PreprocAppBase):
    def __init__(self, *args,
                 recon_param=None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.recon_param = recon_param
    
    def compose(self):
        eiger_zmq_rx, pos_rx, preproc_op = super().compose()
        
        recon = ReconOp(self, param=self.recon_param,
                        postprocessing_flag=False,
                        name="recon")
        batch_stacker = BatchedResultStackerOp(self, name="batch_stacker")
        self.add_flow(preproc_op, recon)
        self.add_flow(recon, batch_stacker)
        
        return eiger_zmq_rx, pos_rx, batch_stacker
        

class PtychoApp(PtychoAppBase):
    def compose(self):
        _, _, batch_stacker = super().compose()
        sink = sink_func(self, name="sink")
        self.add_flow(batch_stacker, sink)


if __name__ == "__main__":
    eiger_ip, eiger_port, msg_format, simulate_position_data_stream, position_data_path = parse_source_args()
    
    recon_param = parse_config('/eiger_dir/ptycho_config',Param())
    recon_param.working_directory = "/eiger_dir/"
    recon_param.gpus = [0]
    recon_param.scan_num = 257331

    app = PtychoApp(
        eiger_ip=eiger_ip,
        eiger_port=eiger_port,
        msg_format=msg_format,
        simulate_position_data_stream=simulate_position_data_stream,
        position_data_path=position_data_path,
        recon_param=recon_param)
    
    scheduler = EventBasedScheduler(
                app,
                worker_thread_number=16,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                name="event_based_scheduler",
            )
    app.scheduler(scheduler)
    
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
    
    
