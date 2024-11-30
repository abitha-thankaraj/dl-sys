import triton
import triton.language as tl
import numpy as np
import cupy as cp

__device_name__ = "triton"
_datatype = np.float32
_datatype_size = np.dtype(_datatype).itemsize

class Array:
    def __init__(self, size):
        self.array = cp.empty(size, dtype=cp.float32)
        self._size = size

    @property
    def size(self):
        return self._size

    # @property
    def ptr(self):
        return self.array.data.ptr


# Interface functions
def to_numpy(a, shape, strides, offset):
    # Create a strided view and convert to numpy
    view = cp.lib.stride_tricks.as_strided(
        a.array[offset:],
        shape=shape,
        strides=tuple(s * a.array.dtype.itemsize for s in strides)
    )
    return cp.asnumpy(view)

def from_numpy(a, out):
    # Copy flattened numpy array to CuPy array
    out.array[:] = cp.asarray(a.flatten())
    cp.cuda.Device().synchronize()

def fill(out, val):
    # Fill array with value
    out.array.fill(val)
    cp.cuda.Device().synchronize()

def compact(a, out, shape, strides, offset):
    # Create strided view and copy to output
    view = cp.lib.stride_tricks.as_strided(
        a.array[offset:],
        shape=shape,
        strides=tuple(s * a.array.dtype.itemsize for s in strides)
    )
    out.array[:] = view.flatten()
    cp.cuda.Device().synchronize()

def ewise_setitem(a, out, shape, strides, offset):
    # Copy array to strided view
    view = cp.lib.stride_tricks.as_strided(
        out.array[offset:],
        shape=shape,
        strides=tuple(s * out.array.dtype.itemsize for s in strides)
    )
    view[:] = a.array.reshape(shape)
    cp.cuda.Device().synchronize()

def scalar_setitem(size, val, out, shape, strides, offset):
    # Copy scalar value to strided view
    view = cp.lib.stride_tricks.as_strided(
        out.array[offset:],
        shape=shape,
        strides=tuple(s * out.array.dtype.itemsize for s in strides)
    )
    view[:] = val
    cp.cuda.Device().synchronize()