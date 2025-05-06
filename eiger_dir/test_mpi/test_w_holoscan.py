import numpy as np
from mpi4py import MPI
from holoscan.core import Application, Operator, OperatorSpec
from holoscan.conditions import CountCondition
import time

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

N_ELEM = int(1e6)

class SrcOp(Operator):

    def setup(self, spec):
        spec.output("out")
    
    def compute(self, op_input, op_output, context):
        output = np.zeros((size, N_ELEM), dtype=np.float32)
        for i in range(size):
            output[i, :] = np.arange(N_ELEM)
        op_output.emit(output, "out")



class ProcessOp(Operator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def setup(self, spec):
        if rank == 0:
            spec.input("in")
        spec.output("out")
    
    def compute(self, op_input, op_output, context):
        start_time = time.time()
        
        if rank == 0:
            input = op_input.receive("in")
        else:
            input = None
        output = np.empty(N_ELEM, dtype=np.float32)
        comm.Scatter(input, output, root=0)
        output += rank
        op_output.emit(output, "out")

        end_time = time.time()
        print(f"ProcessOp End: {rank=}, {self.count=}, duration: {end_time - start_time}")
        self.count += 1


class GatherOp(Operator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def setup(self, spec):
        spec.input("in")
        if rank == 0:
            spec.output("out")

    def compute(self, op_input, op_output, context):
        start_time = time.time()
        
        input = op_input.receive("in")
        output = None
        if rank == 0:
            output = np.empty([size, N_ELEM], dtype=np.float32)
        comm.Gather(input, output, root=0)
        if rank == 0:
            op_output.emit(output, "out")
        
        end_time = time.time()
        print(f"GatherOp End: {rank=}, {self.count=}, duration: {end_time - start_time}")
        self.count += 1

class SinkOp(Operator):

    def setup(self, spec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        input = op_input.receive("in")
        print(f"SinkOp: {rank=}: {input[:, :2]=}, time: {time.time()}")

class TestApp(Application):

    def compose(self):
        if rank == 0:
            src_op = SrcOp(self, CountCondition(self, 2), name="src_op")
        process_op = ProcessOp(self, name="process_op")
        gather_op = GatherOp(self, name="gather_op")
        sink_op = SinkOp(self, name="sink_op")

        if rank == 0:
            self.add_flow(src_op, process_op)
    
        self.add_flow(process_op, gather_op)
        if rank == 0:
            self.add_flow(gather_op, sink_op)


if __name__ == "__main__":
    app = TestApp()
    app.run()



