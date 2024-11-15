"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np
from .nn_basic import Parameter, Module


class Conv(Module):
    """
    Multi-channel 2D convolutional layer
    IMPORTANT: Accepts inputs in NCHW format, outputs also in NCHW format
    Only supports padding=same
    No grouped convolution or dilation
    Only supports square kernels
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, bias=True, device=None, dtype="float32"):
        super().__init__()
        if isinstance(kernel_size, tuple):
            kernel_size = kernel_size[0]
        if isinstance(stride, tuple):
            stride = stride[0]
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride

        ### BEGIN YOUR SOLUTION
        # calculate padding, so that after convolution
        # the shapes remain the same
        self.padding = (kernel_size - 1) // 2

        # define convolution kernel weight
        ## Kaiming Uniform initialization
        convolution_kernel_shape = (kernel_size, kernel_size, in_channels, out_channels)
        self.weight = Parameter(
          init.kaiming_uniform(
            fan_in=None,
            fan_out=None,
            shape=convolution_kernel_shape,
            device=device,
            dtype=dtype,
          )
        )

        if bias:
          bias_higher_bound = 1.0 / np.sqrt(in_channels * kernel_size * kernel_size)
          bias_lower_bound = -bias_higher_bound

          ## Unform init
          self.bias = Parameter(
            init.rand(
              self.out_channels,
              low=bias_lower_bound,
              high=bias_higher_bound,
              dtype=dtype,
              device=device,
            )
          )
        
        else:
          self.bias = None
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # reshape x from (N, C, H, W) to (N, H, W, C)
        x_transposed = x.transpose((1, 2)).transpose((2, 3))

        out = ops.conv(
          a=x_transposed,
          b=self.weight,
          stride=self.stride,
          padding=self.padding,
        )

        if self.bias is not None:
          reshaped_bias = self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)
          out += self.bias.reshape((1, 1, 1, self.out_channels)).broadcast_to(out.shape)

        # reshape to expected output shape (N, C, H, W)
        out = out.transpose((2, 3)).transpose((1, 2))
        return out

        ### END YOUR SOLUTION