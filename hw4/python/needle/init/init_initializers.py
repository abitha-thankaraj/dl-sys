import math
from .init_basic import *


def xavier_uniform(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    a = math.sqrt(6.0 / (fan_in + fan_out)) * gain
    return a * (2 * rand(fan_in, fan_out, **kwargs) - 1)

    ### END YOUR SOLUTION


def xavier_normal(fan_in, fan_out, gain=1.0, **kwargs):
    ### BEGIN YOUR SOLUTION
    std = math.sqrt(2 / (fan_in + fan_out)) * gain
    return std * randn(fan_in, fan_out, **kwargs)    
    ### END YOUR SOLUTION

def kaiming_uniform(fan_in, fan_out, shape=None, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    if shape is not None:
      assert len(shape) == 4
      assert shape[0] == shape[1]
      fan_in = (shape[0] ** 2) * shape[2]
      fan_out = (shape[0] ** 2) * shape[1]
    else:
      shape = (fan_in, fan_out)
    gain = math.sqrt(2)
    bound = math.sqrt(3 / fan_in) * gain
    return bound * (2 * rand(*shape, **kwargs) - 1)        
    ### END YOUR SOLUTION



def kaiming_normal(fan_in, fan_out, nonlinearity="relu", **kwargs):
    assert nonlinearity == "relu", "Only relu supported currently"
    ### BEGIN YOUR SOLUTION
    gain = math.sqrt(2)
    std = gain / math.sqrt(fan_in)
    return std * randn(fan_in, fan_out, **kwargs)
    ### END YOUR SOLUTION