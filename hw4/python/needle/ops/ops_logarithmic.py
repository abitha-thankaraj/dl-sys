from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

from ..backend_selection import array_api, BACKEND 

class LogSoftmax(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes    
    
    def compute(self, Z):
        ### BEGIN YOUR SOLUTION      
        max_Z = array_api.max(Z, axis=self.axes, keepdims=True)
        exp_Z = array_api.exp(Z - max_Z)
        sum_exp_Z = array_api.sum(exp_Z, axis=self.axes, keepdims=True)
        log_sum_exp = array_api.log(sum_exp_Z)
        return Z - max_Z - log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z, = node.inputs
        return out_grad - summation(out_grad, axes=-1).reshape((Z.shape[0], 1)) * exp(node)
        ### END YOUR SOLUTION        

def logsoftmax(a, axes=None):
    return LogSoftmax(axes=(-1,))(a)


class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        ### BEGIN YOUR SOLUTION

        # max_Z = Z.max(axis=self.axes, keepdims=True) 
        # # array_api.max(Z, self.axes, keepdims=True) 
        # return array_api.log(array_api.sum(array_api.exp(Z - max_Z), self.axes)) + array_api.max(Z, self.axes)

        max_z_original = Z.max(axis=self.axes, keepdims=True) 
        max_z_reduce = Z.max(axis=self.axes)
        return array_api.log(array_api.sum(array_api.exp(Z - max_z_original.broadcast_to(Z.shape)), axis=self.axes)) + max_z_reduce

        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        # Z = node.inputs[0]
            
        # if self.axes:
        #     new_shape = [1] * len(Z.shape)
        #     s = set(self.axes)
        #     j = 0
        #     for i in range(len(Z.shape)):
        #         if i not in s:
        #             new_shape[i] = node.shape[j]
        #             j += 1
        #     # print("new_shape", new_shape)
        #     grad_new = reshape(out_grad, new_shape)
        #     node_new = reshape(node, new_shape)
        # # scalar
        # else:
        #     node_new = node
        #     grad_new = out_grad

        # final = grad_new * exp(node.inputs[0] - node_new)
        # return final
# retrieve input data
        Z = node.inputs[0]

        # retrieve value inside tensor
        # calculate max for numerical stability
        max_z = Tensor(Z.realize_cached_data().max(axis=self.axes, keepdims=True), device=Z.device)
        max_z_broadcast = max_z.broadcast_to(Z.shape)

        # calculate exp_Z
        exp_Z = exp(Z - max_z_broadcast)

        # gradient = out_grad * (exp_Z / sum_exp_Z)
        # However, out_grad and sum_exp_Z has the same shape,
        # so we do the division first
        sum_exp_Z = summation(exp_Z, self.axes)
        grad = out_grad / sum_exp_Z

        # Expand out_grad to have the same shape as Z
        # NOTE: copied over from the prior implementation of Summation

        # this is the shape we will use to reshape out_grad to
        # essentially the shape of the input
        new_shape = list(Z.shape)

        # However, the summed over axes will have size = 1
        # along these axes
        if isinstance(self.axes, tuple):
          axes = self.axes

        elif isinstance(self.axes, list):
          axes = tuple(self.axes)

        elif isinstance(self.axes, int):
          axes = (self.axes,)

        # in this case we sum over all the axes and return a single number
        # so need to broadcast it back to the original shape
        elif self.axes is None:
          axes = [i for i in range(len(Z.shape))]

        else:
          raise ValueError("Given axes is not supported!")

        for axis in axes:
          new_shape[axis] = 1

        new_shape = tuple(new_shape)

        # First we reshape to insert new axis into the shape
        grad = reshape(grad, new_shape)

        # Now we broadcast back to input shape
        grad = broadcast_to(grad, Z.shape)

        # multiply with exp_Z to return the output
        return multiply(grad, exp_Z)
        ### END YOUR SOLUTION
def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
