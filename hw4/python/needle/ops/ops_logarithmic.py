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
        max_Z = array_api.max(Z, self.axes, keepdims=True) 
        return array_api.log(array_api.sum(array_api.exp(Z - max_Z), self.axes)) + array_api.max(Z, self.axes)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
            
        if self.axes:
            new_shape = [1] * len(Z.shape)
            s = set(self.axes)
            j = 0
            for i in range(len(Z.shape)):
                if i not in s:
                    new_shape[i] = node.shape[j]
                    j += 1
            # print("new_shape", new_shape)
            grad_new = reshape(out_grad, new_shape)
            node_new = reshape(node, new_shape)
        # scalar
        else:
            node_new = node
            grad_new = out_grad

        final = grad_new * exp(node.inputs[0] - node_new)
        return final

        ### END YOUR SOLUTION
def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
