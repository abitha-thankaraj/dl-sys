# import triton
# import triton.language as tl
# import numpy as np
# import cupy as cp

# __device_name__ = "triton"
# _datatype = np.float32
# _datatype_size = np.dtype(_datatype).itemsize

# class Array:
#     def __init__(self, size):
#         self.array = cp.empty(size, dtype=cp.float32)
#         self._size = size

#     @property
#     def size(self):
#         return self._size

#     def ptr(self):
#         return self.array.data.ptr

# # # Interface functions
# def to_numpy(a, shape, strides, offset):
#     # Create a strided view and convert to numpy
#     view = cp.lib.stride_tricks.as_strided(
#         a.array[offset:],
#         shape=shape,
#         strides=tuple(s * a.array.dtype.itemsize for s in strides)
#     )
#     return cp.asnumpy(view)

# def from_numpy(a, out):
#     # Copy flattened numpy array to CuPy array
#     out.array[:] = cp.asarray(a.flatten())
#     cp.cuda.Device().synchronize()

# def fill(out, val):
#     # Fill array with value
#     out.array.fill(val)
#     cp.cuda.Device().synchronize()


# def compact(a, out, shape, strides, offset):
#     # Create strided view and copy to output
#     # return compact_triton(a, out, shape, strides, offset)
#     # return compact_triton_nd(a, out, shape, strides, offset)
#     view = cp.lib.stride_tricks.as_strided(
#         a.array[offset:],
#         shape=shape,
#         strides=tuple(s * a.array.dtype.itemsize for s in strides)
#     )
#     out.array[:] = view.flatten()
#     cp.cuda.Device().synchronize()


# def ewise_setitem(a, out, shape, strides, offset):
#     # Copy array to strided view
#     view = cp.lib.stride_tricks.as_strided(
#         out.array[offset:],
#         shape=shape,
#         strides=tuple(s * out.array.dtype.itemsize for s in strides)
#     )
#     view[:] = a.array.reshape(shape)
#     cp.cuda.Device().synchronize()

# def scalar_setitem(size, val, out, shape, strides, offset):
#     # Copy scalar value to strided view
#     view = cp.lib.stride_tricks.as_strided(
#         out.array[offset:],
#         shape=shape,
#         strides=tuple(s * out.array.dtype.itemsize for s in strides)
#     )
#     view[:] = val
#     cp.cuda.Device().synchronize()

import torch
import numpy as np
import triton
import triton.language as tl

__device_name__ = "triton"
_datatype = torch.float32
_datatype_size = torch.finfo(_datatype).bits // 8  # Convert bits to bytes

class Array:
    def __init__(self, size):
        self.array = torch.empty(size, dtype=_datatype, device="cuda")

    @property
    def size(self):
        return self.array.numel()

def to_numpy(a, shape, strides, offset):
    base_tensor = a.array[offset:]
    strided_tensor = torch.as_strided(base_tensor, shape, strides).contiguous()
    return strided_tensor.cpu().numpy()

def from_numpy(a, out):
    # Convert numpy array to tensor and copy to GPU
    tensor = torch.from_numpy(a.flatten()).cuda().contiguous()
    out.array.copy_(tensor)

def ewise_add(a, b, out):
    add(a.array, b.array, out.array)

def scalar_add(a, val, out):
    out.array[:] = a.array + val


@triton.jit
def compact_kernel(
    in_ptr,     # Input tensor pointer
    out_ptr,    # Output tensor pointer
    n_elements, # Number of elements
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load and store values
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

def compact(a, out, shape, strides, offset):
    strided_tensor = torch.as_strided(a.array[offset:], shape, strides).contiguous() 
    # strided_tensor = torch.as_strided(a.array[offset:], shape, strides)

    # Then compact using kernel
    n_elements = strided_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    compact_kernel[grid](strided_tensor, out.array, 
                        n_elements, BLOCK_SIZE=1024)


@triton.jit
def add_kernel(x_ptr,  # *Pointer* to first input vector.
               y_ptr,  # *Pointer* to second input vector.
               output_ptr,  # *Pointer* to output vector.
               n_elements,  # Size of the vector.
               BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
               # NOTE: `constexpr` so it can be used as a shape value.
               ):
    pid = tl.program_id(axis=0)  # We use a 1D launch grid so axis is 0.
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor, output: torch.Tensor):
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    return output


# From : https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c

# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1
    # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
    # by to get the element one row down (A has M rows).
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    # -----------------------------------------------------------
    # Map program ids `pid` to the block of C it should compute.
    # This is done in a grouped ordering to promote L2 data reuse.
    # See above `L2 Cache Optimizations` section for details.
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetics` section for details
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # -----------------------------------------------------------
    # Iterate to compute a block of the C matrix.
    # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
    # of fp32 values for higher accuracy.
    # `accumulator` will be converted back to fp16 after the loop.
    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        # Load the next block of A and B, generate a mask by checking the K dimension.
        # If it is out of bounds, set it to 0.
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        # We accumulate along the K dimension.
        accumulator += tl.dot(a, b)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float16)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
@triton.jit
def leaky_relu(x):
    x = x + 1
    return tl.where(x >= 0, x, 0.01 * x)

def matmul(a, b, out, m, n, p):
    # Check constraints.
    # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    # assert a.is_contiguous(), "Matrix A must be contiguous"
    # assert b.is_contiguous(), "Matrix B must be contiguous"
    # m, n = a.shape
    # n, p = b.shape
    # Allocates output.
    c = torch.empty((m, p), device=a.device, dtype=a.dtype)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (
        triton.cdiv(m, META['BLOCK_SIZE_M']) * triton.cdiv(p, META['BLOCK_SIZE_N']),
    )
    matmul_kernel[grid](
        a, b, out,
        m, p, n,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        ACTIVATION=""
    )
    return c