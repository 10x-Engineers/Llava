from max.engine import InputSpec, InferenceSession
from python import Python
from utils.index import Index
from time import now
from max.graph import Graph, TensorType, Type, ops, Symbol
from max import engine
from max.tensor import Tensor, TensorShape
from max.engine import Model
from algorithm import sum
from utils.numerics import inf
from algorithm import parallelize
from max.graph.symbol import SymbolicSlice
from utils import StaticIntTuple
from buffer.dimlist import Dim
from random import seed
from max.graph.checkpoint import load


# fn main() raises:
    # var graph2 = Graph(in_types=List[Type](TensorType(DType.float32, "1", "b", "c"),TensorType(DType.float32, "c"), TensorType(DType.float32, "c")))
    # var mean = ops.layer_norm(graph2[0],gamma = graph2[1], beta = graph2[2] , epsilon = 1e-5)
    # graph2.output(mean)
    # graph2.verify()
    # var session = engine.InferenceSession()
    # var norm = session.load(graph2)


    # var t = Tensor[DType.float32] (2,2)
    # for i in range(4):
    #     for j in range(3):
    #         t[Index(i,j)] = i+j
    # print(t)
    # var t1 = Tensor[DType.int32] (1)
    # t1[0] = 2
    # print(t1)
    # var x = Slice(2,2)
    # var x = List[Slice] (Slice(2,2))
    # var x1 = List[SymbolicSlice] ()
    # var od = List[Dim] (Dim(2), Dim(2))

    # var graph2 = Graph(in_types=List[Type](TensorType(DType.float32, 2, 2)))
    # var mean = ops.slice(input = graph2[0], slices = x1, out_dims = od)
    # graph2.output(mean)
    # graph2.verify()
    # var session = engine.InferenceSession()
    # var split = session.load(graph2)

    # var results = split.execute("input0", t, "input2", t1)
    # var out = results.get[DType.float32]("output0")
    # print(out)

    # var x = slice(0,3)

    # var s = "My name is khan"
    # print(x,s)
    
    # print(x.__str__())

    # var y = Tensor[DType.float32] (3)
    # print(y)

    # print(y[1])

    # var hurray = Tensor[DType.float32] (4)
    # var msg: String = "Hello Mojo"

    # # Both are equivalent and print "Mojo".
    # print(msg[6:])
    # print(msg.__getitem__(Slice(6, len(msg))))


    # var t = Tensor[DType.float32] (3,1,16,792,72)
    # print(t.num_elements())

    # var t1 = Tensor[DType.float32] (1,16,792,72)
    # print(t1.num_elements())

    # t1.store(0,t.load[width = 912384] (0))

    # print(t1)



# fn test():
#     seed(43)
#     var t = Tensor[DType.float32].randn((2, 2))
#     print(t)

#     print(t.load[width=1](0), t.load[width=1](1))

#     # fn p(ten: Tensor[DType.float32], *indices: Int):
#     #     print(ten.load[width=2](indices))

#     # p(t, 0, 1)

#     # print(t.load[1](0), t.load[1](1))

# def main():
#     test()

# fn main() raises:
#     var qkv = Tensor[DType.float32].randn((3,1,16,792,72))
#     print(qkv)
#     print(qkv.num_elements())

#     var q = Tensor[DType.float32] (1,16,792,72)
#     print(q.num_elements())

#     alias load_size = 256
#     var start_q = 0
#     var start_qkv = 0
#     var q_num_elements = q.num_elements()

#     for i in range(0, q_num_elements, load_size):
#         q.store(start_q, qkv.load[width = load_size](start_qkv))
#         start_q += load_size
#         start_qkv += load_size

#     print(q)

# alias load_size2 = 128
# fn main() raises:

#     var graph13 = Graph(in_types=List[Type](TensorType(DType.float32, "a", "b")))
#     transposed = ops.transpose(graph13[0], 0, 1)
#     graph13.output(transposed)
#     graph13.verify()
#     var session = engine.InferenceSession()
#     var transpose_01 = session.load(graph13)

#     var full_img_features = Tensor[DType.float32].randn((1,729,1152))
#     print("full_img_features:\n", full_img_features)

#     var s:TensorShape = (729,1152)
#     var full_img_features_reshaped = Tensor[DType.float32](s)
#     var q_num_elements = full_img_features_reshaped.num_elements()
#     var start_q = 0
#     for i in range(0, q_num_elements, load_size2):
#         full_img_features_reshaped.store(start_q, full_img_features.load[width=load_size2](start_q))
#         start_q += load_size2

#     print("full_img_features_reshaped:\n", full_img_features_reshaped)

#     var results = transpose_01.execute("input0", full_img_features_reshaped)
#     var t = results.get[DType.float32]("output0")
#     print("t:\n", t)

#     s = (1152,27,27)
#     var r = t.reshape(s)
#     print("reshaped_patch_features_1:\n", r)

fn main() raises:
    var tensors = load("encoder_output.maxckpt")
    var x = tensors.get[DType.float32]("x")

    print(x)