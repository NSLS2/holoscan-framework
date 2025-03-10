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

from pipeline_source import parse_source_args
from pipeline_preprocess import PreprocAppBase

import nvtx

class ReconOp(Operator):
    def __init__(self, *args, param=None, **kwargs):
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
                prb_init = deal_with_init_prb(self.recon, self.param, diff_d_to_add)
                self.recon.prb += cp.asnumpy(prb_init.astype(self.recon.complex_precision))
                self.recon.prb_d[:, :] += prb_init.astype(self.recon.complex_precision)

        with nvtx.annotate("self.recon.update_arrays", color="green"):
            self.recon.update_arrays(diff_d_to_add, points_to_add * -1)

        self.recon.recon_ptycho_run()
        output = self.recon.fetch_obj_ave()

        op_output.emit(output, "result")


class ReconResultOp(Operator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.logger = logging.getLogger("ReconResultOp")
        logging.basicConfig(level=logging.INFO)
        self.batched_result = None
        # self.counter = 0
        
    def setup(self, spec: OperatorSpec):
        spec.input("in")
        spec.output("out")
        
    def compute(self, op_input, op_output, context):
        obj = op_input.receive("in")
        if self.batched_result is None:
            self.batched_result = obj.copy()
        else:
            # this implies that at most two batches overlap
            self.batched_result = np.nanmean(
                np.array([self.batched_result, obj.copy()]), axis=0)
        # self.counter += 1
        # self.batched_result.append(obj)
        # bla = np.array(self.batched_result)
        # out = np.nanmean(bla, axis=0)
        
        # average = self.batched_result / self.counter
        out = np.nan_to_num(self.batched_result, nan=0.0)
        op_output.emit(out, "out")


@create_op(inputs="result")
def sink_func(result):
    print(f"SinkOp received the result from reconstruction {np.sum(result>0)=}")


class PtychoAppBase(PreprocAppBase):
    def __init__(self, *args,
                 recon_param=None,
                 num_parallel_streams=1,  # Match default from PreprocAppBase
                 **kwargs):
        super().__init__(*args, num_parallel_streams=num_parallel_streams, **kwargs)
        self.recon_param = recon_param
    
    def compose(self):
        eiger_zmq_rx, pos_rx, gather_op = super().compose()
        
        # Create N reconstruction operators
        recon_ops = []
        for i in range(1, self.num_parallel_streams + 1):
            recon = ReconOp(self, 
                           param=self.recon_param,
                           postprocessing_flag=False,
                           name=f"recon{i}")
            recon_ops.append(recon)
            # Connect gather_op output to this recon operator
            self.add_flow(gather_op, recon, {(f"batch{i}", "batch")})
        
        # Create single batch stacker for combining all results
        recon_result_stacker = ReconResultOp(self, name="recon_result_stacker")
        
        # Connect all recon operators to the same batch stacker
        for recon in recon_ops:
            self.add_flow(recon, recon_result_stacker, {("result", "in")})
        
        return eiger_zmq_rx, pos_rx, recon_result_stacker
        

class PtychoApp(PtychoAppBase):
    def compose(self):
        _, _, recon_result_stacker = super().compose()
        sink = sink_func(self, name="sink")
        self.add_flow(recon_result_stacker, sink)


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
        num_parallel_streams=1,
        num_batches_per_emit=2,
        num_batches_overlap=0,
        simulate_position_data_stream=simulate_position_data_stream,
        position_data_path=position_data_path,
        recon_param=recon_param)
    
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
                worker_thread_number=16,
                check_recession_period_ms=0.001,
                stop_on_deadlock=True,
                stop_on_deadlock_timeout=500,
                name="multithread_scheduler",
            )
    app.scheduler(scheduler)
    
    app.run()
    
    
