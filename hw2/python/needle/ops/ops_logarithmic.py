from typing import Optional
from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp

from .ops_mathematic import *

import numpy as array_api

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
        maxz = array_api.max(Z, axis=self.axes, keepdims=1)
        log_sum_exp = array_api.log(array_api.exp(Z - maxz).sum(axis=self.axes, keepdims=1)) + maxz
        if self.axes:
            out_shape = [size for i, size in enumerate(Z.shape) if i not in self.axes]
        else:
            # scalar
            out_shape = ()
        log_sum_exp.resize(tuple(out_shape))
        return log_sum_exp
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        Z = node.inputs[0]
        
        if self.axes:
            # Create a shape with 1s in the axes that were reduced
            shape = [1 if i in self.axes else size for i, size in enumerate(Z.shape)]
            # Reshape node and gradient to match this shape for broadcasting
            node_reshaped = node.reshape(shape)
            out_grad_reshaped = out_grad.reshape(shape)
        else:
            # No need to reshape if no axes were reduced
            node_reshaped = node
            out_grad_reshaped = out_grad

        # Gradient of log-sum-exp is exp(Z - node) * out_grad
        return out_grad_reshaped * exp(Z - node_reshaped)
        ### END YOUR SOLUTION


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)

