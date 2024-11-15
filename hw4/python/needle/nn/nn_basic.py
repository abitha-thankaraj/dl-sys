"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []


class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


# class Linear(Module):
#     def __init__(
#         self, in_features, out_features, bias=True, device=None, dtype="float32"
#     ):
#         super().__init__()
#         self.in_features = in_features
#         self.out_features = out_features

#         ### BEGIN YOUR SOLUTION
#         self.use_bias = bias
#         self.weight = Parameter(
#           init.kaiming_uniform(
#             fan_in=self.in_features, 
#             fan_out=self.out_features,
#             dtype=dtype,
#             requires_grad=True,
#             device=device,
#           )
#         )

#         if self.use_bias:
#           self.bias = Parameter(
#             init.kaiming_uniform(
#               fan_in=self.out_features, 
#               fan_out=1,
#               dtype=dtype,
#               requires_grad=True,
#               device=device,
#             ).transpose()
#           )
#         ### END YOUR SOLUTION

#     def forward(self, X: Tensor) -> Tensor:
#         ### BEGIN YOUR SOLUTION
#         y = X.matmul(self.weight)
#         if self.use_bias:
#           y += self.bias.broadcast_to(y.shape)
#         return y
#         ### END YOUR SOLUTION

class Linear(Module):
    def __init__(
        self, in_features, out_features, bias=True, device=None, dtype="float32"
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(init.kaiming_uniform(in_features, out_features, device=device, dtype=dtype))
            # Add transpose to make it compatible with the forward pass
        self.bias = Parameter(init.kaiming_uniform(out_features, 1, device=device, dtype=dtype).transpose()) if bias else None
        ### END YOUR SOLUTION

    def forward(self, X: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        # X : mxn; weight: nxp; bias: 1xp
        out = X.matmul(self.weight)
        if self.bias:
            out += ops.broadcast_to(self.bias, (*X.shape[:-1], self.out_features))
        return out
        ### END YOUR SOLUTION


class Flatten(Module):
    def forward(self, X):
        ### BEGIN YOUR SOLUTION
        B = X.shape[0]
        N = 1
        for i in range(1, len(X.shape)):
          N = N * X.shape[i]

        newshape = (B, N)
        return ops.reshape(a=X, shape=newshape)
        ### END YOUR SOLUTION


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return ops.relu(x)
        ### END YOUR SOLUTION

class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        out = None
        
        for module in self.modules:
          if out is None:
            out = module(x)
          else:
            out = module(out)

        return out
        ### END YOUR SOLUTION


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        ### BEGIN YOUR SOLUTION
        batch_size, N = logits.shape[0], logits.shape[1]
        one_hot_y = init.one_hot(n=N, i=y, device=logits.device)

        first_term = ops.logsumexp(a=logits, axes=(1,))
        z_y = logits * one_hot_y
        z_y = ops.summation(a=z_y, axes=(1,))
        difference = first_term - z_y

        # average over the batch
        return ops.summation(difference) / batch_size
        ### END YOUR SOLUTION


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = Parameter(
          init.ones(
            dim, 
            dtype=dtype, 
            device=device, 
            requires_grad=True,
          )
        )

        self.bias = Parameter(
          init.zeros(
            dim, 
            dtype=dtype, 
            device=device, 
            requires_grad=True,
          )
        )

        self.running_mean = init.zeros(
          dim,
          dtype=dtype, 
          device=device, 
          requires_grad=False,
        )

        self.running_var = init.ones(
          dim,
          dtype=dtype, 
          device=device, 
          requires_grad=False,
        )
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION

        if self.training:
          mean = x.sum((0,)) / x.shape[0]
          mean_reshaped = mean.reshape((1, x.shape[1]))

          # broadcast to the shape of x, to calculate variance
          mean_broadcast = mean_reshaped.broadcast_to(x.shape)
          var = ((x - mean_broadcast) ** 2).sum((0,)) / x.shape[0]

          # reshape variance too
          var_reshaped = var.reshape((1, x.shape[1]))
          var_broadcast = var_reshaped.broadcast_to(x.shape)

          # calculate running mean/variance
          self.running_mean = self.momentum * mean + (1.0 - self.momentum) * self.running_mean
          self.running_var = self.momentum * var + (1.0 - self.momentum) * self.running_var

        else:
          # not training, so we use running mean/variance for doing things
          # reshape running mean/variance
          mean_broadcast = self.running_mean.reshape((1, x.shape[1])).broadcast_to(x.shape)
          var_broadcast = self.running_var.reshape((1, x.shape[1])).broadcast_to(x.shape)

        weight_reshaped = self.weight.reshape((1, x.shape[1])).broadcast_to(x.shape)
        bias_reshaped = self.bias.reshape((1, x.shape[1])).broadcast_to(x.shape)

        # calculate normalized data, multiply by weight and add bias
        normalized_data = (x - mean_broadcast) / (var_broadcast + self.eps) ** 0.5
        return weight_reshaped * normalized_data + bias_reshaped

        ### END YOUR SOLUTION

class BatchNorm2d(BatchNorm1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x: Tensor):
        # nchw -> nhcw -> nhwc
        s = x.shape
        _x = x.transpose((1, 2)).transpose((2, 3)).reshape((s[0] * s[2] * s[3], s[1]))
        y = super().forward(_x).reshape((s[0], s[2], s[3], s[1]))
        return y.transpose((2,3)).transpose((1,2))


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.w = Parameter(init.ones(dim, device=device, dtype=dtype))
        self.b = Parameter(init.zeros(dim, device=device, dtype=dtype))
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean = ops.summation(x, axes=(1,)).reshape((x.shape[0], 1)) / x.shape[1]
        mean = ops.broadcast_to(mean, x.shape)
        var = ops.summation((x - mean) ** 2, axes=(1,)).reshape((x.shape[0], 1)) / x.shape[1]
        var = ops.broadcast_to(var, x.shape)

        x_hat = (x - mean) / ops.power_scalar(var + self.eps, 0.5)

        broadcast_w = ops.broadcast_to(self.w, x.shape)
        broadcast_b = ops.broadcast_to(self.b, x.shape)

        return broadcast_w * x_hat + broadcast_b
        ## END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
          # randb is implemented such that
          # p is the probability of getting 1 (i.e., bernoulli parameter)
          # However, self.p for dropout is the probability of zeroing out
          dropout_mask = init.randb(
            *x.shape,
            p=(1.0 - self.p),
            device=x.device,
            dtype=x.dtype,
          )
          out = x * dropout_mask / (1.0 - self.p)

        else:
          out = x
        return out
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION
