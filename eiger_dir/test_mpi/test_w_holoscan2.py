# Pipeline:
# rank 0
# Src --> Scatter 
#            A                      
#            |                      
#            |  ===  MPI COMM  ===  
#            |                      
# rank 1     V                     
#         Scatter --> Process --> Gather --> Sink
#            A                      A
#            |                      |
#            |  ===  MPI COMM  ===  |  
#            |                      |
# rank 2     V                      V
#         Scatter --> Process --> Gather
#
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
        output = np.zeros((size - 1, N_ELEM), dtype=np.float32)
        for i in range(size - 1):
            output[i, :] = np.arange(N_ELEM)
        op_output.emit(output, "out")



class ScatterOp(Operator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def setup(self, spec):
        if rank == 0:
            spec.input("in")
        else:
            spec.output("out")
    
    def compute(self, op_input, op_output, context):
        start_time = time.time()
        
        if rank == 0:
            # On rank 0, receive the input and send to other ranks
            input = op_input.receive("in")
            for i in range(size-1):
                comm.Send(input[i], dest=i+1, tag=0) # this can be also done with Scatterv
        else:
            # On other ranks, receive data
            output = np.empty(N_ELEM, dtype=np.float32)
            comm.Recv(output, source=0, tag=0)
            op_output.emit(output, "out")

        end_time = time.time()
        print(f"ScatterOp End: {rank=}, {self.count=}, duration: {end_time - start_time}")
        self.count += 1

class ProcessOp(Operator):

    def setup(self, spec):
        spec.input("in")
        spec.output("out")

    def compute(self, op_input, op_output, context):
        input = op_input.receive("in")
        output = input + rank
        op_output.emit(output, "out")

class GatherOp(Operator):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.count = 0

    def setup(self, spec):
        spec.input("in")
        if rank == 1:
            spec.output("out")

    def compute(self, op_input, op_output, context):
        start_time = time.time()
        
        # All other ranks receive their input
        input = op_input.receive("in")
        
        if rank == 1:
            # On rank 1, receive data from ranks 2 to size-1
            output = np.empty([size-1, N_ELEM], dtype=np.float32)
            output[0] = input
            for i in range(2, size):
                comm.Recv(output[i-1], source=i, tag=0) # this can be also done with Scatterv
            op_output.emit(output, "out")
        else:
            # On ranks 2 to size-1, send data to rank 1
            comm.Send(input, dest=1, tag=0)
        
        end_time = time.time()
        print(f"GatherOp End: {rank=}, {self.count=}, duration: {end_time - start_time}")
        self.count += 1

class SinkOp(Operator):

    def setup(self, spec):
        spec.input("in")

    def compute(self, op_input, op_output, context):
        input = op_input.receive("in")
        print(f"SinkOp: {rank=}: {input[:, :2]=}")

class TestApp(Application):

    def compose(self):
        if rank == 0:
            src_op = SrcOp(self, CountCondition(self, 2), name="src_op")
            scatter_op = ScatterOp(self, name="scatter_op")
            self.add_flow(src_op, scatter_op)
        else:
            scatter_op = ScatterOp(self, name="scatter_op")
            process_op = ProcessOp(self, name="process_op")
            gather_op = GatherOp(self, name="gather_op")
            self.add_flow(scatter_op, process_op)
            self.add_flow(process_op, gather_op)

        if rank == 1:
            sink_op = SinkOp(self, name="sink_op")
            self.add_flow(gather_op, sink_op)

    


if __name__ == "__main__":
    app = TestApp()
    app.run()


