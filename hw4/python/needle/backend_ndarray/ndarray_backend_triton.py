import torch
import numpy as np
import triton
import typing as tp
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
    
    def ptr(self):
        return self.array.data_ptr()

def to_numpy(a, shape, strides, offset):
    base_tensor = a.array[offset:]
    strided_tensor = torch.as_strided(base_tensor, shape, strides).contiguous()
    return np.ascontiguousarray(strided_tensor.cpu().numpy())

def from_numpy(a, out):
    # Convert numpy array to tensor and copy to GPU
    tensor = torch.from_numpy(a.flatten()).cuda().contiguous()
    out.array.copy_(tensor)

def ewise_add(a, b, out):
    # add(a.array, b.array, out.array)
    ewise_op(a.array, b.array, out.array, operation="add")

def scalar_add(a, val, out):
    scalar_op(a.array, val, out.array, operation="add")

def ewise_mul(a, b, out):
    ewise_op(a.array, b.array, out.array, operation="multiply")

def scalar_mul(a, val, out):
    scalar_op(a.array, val, out.array, operation="multiply")

def ewise_div(a, b, out):
    ewise_op(a.array, b.array, out.array, operation="divide")

def scalar_div(a, val, out):
    if val == 0:
        raise Exception("Division by zero")
    scalar_op(a.array, val, out.array, operation="divide")

def ewise_sub(a, b, out):
    # out.array[:] = a.array - b.array
    ewise_op(a.array, b.array, out.array, operation="subtract")

def scalar_sub(a, val, out):
    scalar_op(a.array, val, out.array, operation="subtract")

def ewise_eq(a, b, out):
    ewise_op(a.array, b.array, out.array, operation="equals")

def scalar_eq(a, val, out):
    scalar_op(a.array, val, out.array, operation="equals")

def ewise_ge(a, b, out):
    ewise_op(a.array, b.array, out.array, operation="greater_equal")

def scalar_ge(a, val, out):
    scalar_op(a.array, val, out.array, operation="greater_equal")

def scalar_power(a, val, out):
    scalar_op(a.array, val, out.array, operation="power")

def ewise_maximum(a, b, out):
    ewise_op(a.array, b.array, out.array, operation="maximum")
def scalar_maximum(a, val, out):
    scalar_op(a.array, val, out.array, operation="maximum") 

def ewise_log(a, out):
    ewise_unary_op(a.array, out.array, operation="log")
def ewise_exp(a, out):
    ewise_unary_op(a.array, out.array, operation="exp")
def ewise_tanh(a, out):
    ewise_unary_op(a.array, out.array, operation="tanh")

def matmul(a, b, out, m, n, p):
    matmul_(a.array.reshape(m,n), b.array.reshape(n,p), out.array.reshape(m,p))


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
    x = tl.load(in_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x, mask=mask)

def compact(a, out, shape, strides, offset):
    strided_tensor = torch.as_strided(a.array[offset:], shape, strides).contiguous() 
    n_elements = strided_tensor.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    compact_kernel[grid](strided_tensor, out.array, 
                        n_elements, BLOCK_SIZE=1024)

@triton.jit
def ewise_op_kernel(
    x_ptr,
    y_ptr,
    output_ptr,
    n_elements,
    op_type,  # 0: add, 1: multiply, 2: divide, 3: subtract
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the operation based on op_type
    output = tl.where(op_type == 0, x + y,
             tl.where(op_type == 1, x * y,
             tl.where(op_type == 2, x / y,
             tl.where(op_type == 3, x - y,
             tl.where(op_type == 4, x == y,
             tl.where(op_type == 5, x > y,
             tl.where(op_type == 6, x < y,
             tl.where(op_type == 7, x >= y,
             tl.where(op_type == 8, x <= y,
             tl.where(op_type == 10, tl.maximum(x, y),
                     x))))))))))  # default case is identity
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)


@triton.jit
def ewise_unary_op_kernel(
    x_ptr,
    output_ptr,
    n_elements,
    op_type,  # 0: exp, 1: log, 2: tanh
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform the operation based on op_type
    output = tl.where(op_type == 0, tl.exp(x),
                tl.where(op_type == 1, tl.log(x),
                tl.where(op_type == 2, tl.exp(x) + tl.exp(-x) / (tl.exp(x) + tl.exp(-x)),
                     x )))  # default case is subtraction
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def ewise_unary_op(x: torch.Tensor, output:torch.Tensor, operation: str = "exp") -> torch.Tensor:
    """
    Perform unary operation on a tensor using Triton.
    
    Args:
        x: Input tensor (must be on GPU)
        output: Output tensor (must be on GPU)
        operation: One of "exp", "log", "tanh"
        
    Returns:
        Result tensor
    """
    assert x.is_cuda and output.is_cuda, "Input tensors must be on GPU"
    assert x.shape == output.shape, "Input tensors must have the same shape"
    
    # Map operation string to integer code
    op_map = {
        "exp": 0,
        "log": 1,
        "tanh": 2
    }
    op_code = op_map.get(operation.lower(), 0)  # default to exp if unknown operation
    
    # Launch kernel
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    ewise_unary_op_kernel[grid](x, output, n_elements, op_code, BLOCK_SIZE=1024)  
    return output


def ewise_op(x: torch.Tensor, y: torch.Tensor, output:torch.Tensor, operation: str = "add") -> torch.Tensor:
    """
    Perform binary operation on two tensors using Triton.
    
    Args:
        x: First input tensor (must be on GPU)
        y: Second input tensor (must be on GPU)
        output: Output tensor (must be on GPU)
        operation: One of "add", "multiply", "divide", "subtract"
        
    Returns:
        Result tensor
    """
    assert x.is_cuda and y.is_cuda and output.is_cuda, "Input tensors must be on GPU"
    assert x.shape == y.shape ==output.shape, "Input tensors must have the same shape"
    op_map = {
        "add": 0,
        "multiply": 1,
        "divide": 2,
        "subtract": 3,
        "equals": 4,
        "greater": 5,
        "less": 6,
        "greater_equal": 7,
        "less_equal": 8,
        "maximum": 10
 }
    op_code = op_map.get(operation.lower(), 0)  # default to add if unknown operation
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    ewise_op_kernel[grid](x, y, output, n_elements, op_code, BLOCK_SIZE=1024)  
    return output

@triton.jit
def scalar_op_kernel(
    x_ptr,
    y_scalar,
    output_ptr,
    n_elements,
    op_type,  # 0: add, 1: multiply, 2: divide, 3: subtract
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load the inputs
    x = tl.load(x_ptr + offsets, mask=mask)
    y = y_scalar
    # y = tl.load(y_ptr + offsets, mask=mask)
    
    # Perform the operation based on op_type
    output = tl.where(op_type == 0, x + y,
             tl.where(op_type == 1, x * y,
             tl.where(op_type == 2, x / y,
             tl.where(op_type == 3, x - y,
             tl.where(op_type == 4, x == y,
             tl.where(op_type == 5, x > y,
             tl.where(op_type == 6, x < y,
             tl.where(op_type == 7, x >= y,
             tl.where(op_type == 8, x <= y,
            tl.where(op_type == 9, tl.exp(tl.log(x) * y),
            tl.where(op_type == 10, tl.maximum(x, y),
                     x )))))))))))  # default case is identity
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def scalar_op(x: torch.Tensor, y:tp.Any , output:torch.Tensor, operation: str = "add") -> torch.Tensor:
    """
    Perform binary operation on two tensors using Triton.
    
    Args:
        x: First input tensor (must be on GPU)
        y: value 
        output: Output tensor (must be on GPU)
        operation: One of "add", "multiply", "divide", "subtract"
        
    Returns:
        Result tensor
    """
    assert x.is_cuda and output.is_cuda, "Input tensors must be on GPU"
    assert x.shape == output.shape, "Input tensors must have the same shape"
    
    # # Create output tensor
    # output = torch.empty_like(x)
    
    # Map operation string to integer code
    op_map = {
        "add": 0,
        "multiply": 1,
        "divide": 2,
        "subtract": 3,
        "equals": 4,
        "greater": 5,
        "less": 6,
        "greater_equal": 7,
        "less_equal": 8,
        "power": 9,
        "maximum": 10

    }
    op_code = op_map.get(operation.lower(), 0)  # default to add if unknown operation
    
    # Launch kernel
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    scalar_op_kernel[grid](x, y, output, n_elements, op_code, BLOCK_SIZE=1024)  
    return output
##################

"""
Matrix Multiplication
=====================
In this tutorial, you will write a very short high-performance FP16 matrix multiplication kernel that achieves
performance on par with cuBLAS or rocBLAS.

You will specifically learn about:

* Block-level matrix multiplications.

* Multi-dimensional pointer arithmetic.

* Program re-ordering for improved L2 cache hit rate.

* Automatic performance tuning.

"""

# %%
# Motivations
# -----------
#
# Matrix multiplications are a key building block of most modern high-performance computing systems.
# They are notoriously hard to optimize, hence their implementation is generally done by
# hardware vendors themselves as part of so-called "kernel libraries" (e.g., cuBLAS).
# Unfortunately, these libraries are often proprietary and cannot be easily customized
# to accommodate the needs of modern deep learning workloads (e.g., fused activation functions).
# In this tutorial, you will learn how to implement efficient matrix multiplications by
# yourself with Triton, in a way that is easy to customize and extend.
#
# Roughly speaking, the kernel that we will write will implement the following blocked
# algorithm to multiply a (M, K) by a (K, N) matrix:
#
#  .. code-block:: python
#
#    # Do in parallel
#    for m in range(0, M, BLOCK_SIZE_M):
#      # Do in parallel
#      for n in range(0, N, BLOCK_SIZE_N):
#        acc = zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=float32)
#        for k in range(0, K, BLOCK_SIZE_K):
#          a = A[m : m+BLOCK_SIZE_M, k : k+BLOCK_SIZE_K]
#          b = B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]
#          acc += dot(a, b)
#        C[m : m+BLOCK_SIZE_M, n : n+BLOCK_SIZE_N] = acc
#
# where each iteration of the doubly-nested for-loop is performed by a dedicated Triton program instance.

# %%
# Compute Kernel
# --------------
#
# The above algorithm is, actually, fairly straightforward to implement in Triton.
# The main difficulty comes from the computation of the memory locations at which blocks
# of :code:`A` and :code:`B` must be read in the inner loop. For that, we need
# multi-dimensional pointer arithmetic.
#
# Pointer Arithmetic
# ~~~~~~~~~~~~~~~~~~~
#
# For a row-major 2D tensor :code:`X`, the memory location of :code:`X[i, j]` is given
# by :code:`&X[i, j] = X + i*stride_xi + j*stride_xj`.
# Therefore, blocks of pointers for :code:`A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K]` and
# :code:`B[k : k+BLOCK_SIZE_K, n : n+BLOCK_SIZE_N]` can be defined in pseudo-code as:
#
#  .. code-block:: python
#
#    &A[m : m+BLOCK_SIZE_M, k:k+BLOCK_SIZE_K] =  a_ptr + (m : m+BLOCK_SIZE_M)[:, None]*A.stride(0) + (k : k+BLOCK_SIZE_K)[None, :]*A.stride(1);
#    &B[k : k+BLOCK_SIZE_K, n:n+BLOCK_SIZE_N] =  b_ptr + (k : k+BLOCK_SIZE_K)[:, None]*B.stride(0) + (n : n+BLOCK_SIZE_N)[None, :]*B.stride(1);
#
# Which means that pointers for blocks of A and B can be initialized (i.e., :code:`k=0`) in Triton as the following
# code. Also note that we need an extra modulo to handle the case where :code:`M` is not a multiple of
# :code:`BLOCK_SIZE_M` or :code:`N` is not a multiple of :code:`BLOCK_SIZE_N`, in which case we can pad the data with
# some useless values, which will not contribute to the results. For the :code:`K` dimension, we will handle that later
# using masking load semantics.
#
#  .. code-block:: python
#
#    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#    offs_k = tl.arange(0, BLOCK_SIZE_K)
#    a_ptrs = a_ptr + (offs_am[:, None]*stride_am + offs_k [None, :]*stride_ak)
#    b_ptrs = b_ptr + (offs_k [:, None]*stride_bk + offs_bn[None, :]*stride_bn)
#
# And then updated in the inner loop as follows:
#
#  .. code-block:: python
#
#    a_ptrs += BLOCK_SIZE_K * stride_ak;
#    b_ptrs += BLOCK_SIZE_K * stride_bk;
#
#
# L2 Cache Optimizations
# ~~~~~~~~~~~~~~~~~~~~~~
#
# As mentioned above, each program instance computes a :code:`[BLOCK_SIZE_M, BLOCK_SIZE_N]`
# block of :code:`C`.
# It is important to remember that the order in which these blocks are computed does
# matter, since it affects the L2 cache hit rate of our program, and unfortunately, a
# simple row-major ordering
#
#  .. code-block:: Python
#
#    pid = tl.program_id(axis=0)
#    grid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    pid_m = pid // grid_n
#    pid_n = pid % grid_n
#
# is just not going to cut it.
#
# One possible solution is to launch blocks in an order that promotes data reuse.
# This can be done by 'super-grouping' blocks in groups of :code:`GROUP_M` rows before
# switching to the next column:
#
#  .. code-block:: python
#
#    # Program ID
#    pid = tl.program_id(axis=0)
#    # Number of program ids along the M axis
#    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#    # Number of programs ids along the N axis
#    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#    # Number of programs in group
#    num_pid_in_group = GROUP_SIZE_M * num_pid_n
#    # Id of the group this program is in
#    group_id = pid // num_pid_in_group
#    # Row-id of the first program in the group
#    first_pid_m = group_id * GROUP_SIZE_M
#    # If `num_pid_m` isn't divisible by `GROUP_SIZE_M`, the last group is smaller
#    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#    # *Within groups*, programs are ordered in a column-major order
#    # Row-id of the program in the *launch grid*
#    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
#    # Col-id of the program in the *launch grid*
#    pid_n = (pid % num_pid_in_group) // group_size_m
#
# For example, in the following matmul where each matrix is 9 blocks by 9 blocks,
# we can see that if we compute the output in row-major ordering, we need to load 90
# blocks into SRAM to compute the first 9 output blocks, but if we do it in grouped
# ordering, we only need to load 54 blocks.
#
#   .. image:: grouped_vs_row_major_ordering.png
#
# In practice, this can improve the performance of our matrix multiplication kernel by
# more than 10\% on some hardware architecture (e.g., 220 to 245 TFLOPS on A100).
#

# %%
# Final Result
# ------------

import torch

import triton
import triton.language as tl


def is_cuda():
    return triton.runtime.driver.active.get_current_target().backend == "cuda"


# def is_hip_mi200():
#     target = triton.runtime.driver.active.get_current_target()
#     return target.backend == 'hip' and target.arch == 'gfx90a'


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        # Good config for fp8 inputs.
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 128, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4)
    ]


def get_hip_autotune_config():
    return [
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 256, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 16, 'GROUP_SIZE_M': 4, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 2},
            num_warps=8, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8, 'waves_per_eu': 3},
            num_warps=4, num_stages=2),
        triton.Config(
            {'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 1, 'waves_per_eu': 8},
            num_warps=4, num_stages=2),
    ]


def get_autotune_config():
    if is_cuda():
        return get_cuda_autotune_config()
    # else:
    #     return get_hip_autotune_config()


# `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
#   - A list of `triton.Config` objects that define different configurations of
#       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
#   - An auto-tuning *key* whose change in values will trigger evaluation of all the
#       provided configs

#TODO
@triton.autotune(
    configs=get_autotune_config(),
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
        stride_am, stride_ak,  #
        stride_bk, stride_bn,  #
        stride_cm, stride_cn,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
        ACTIVATION: tl.constexpr  #
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
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # ----------------------------------------------------------
    # Create pointers for the first blocks of A and B.
    # We will advance this pointer as we move in the K direction
    # and accumulate
    # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
    # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
    # See above `Pointer Arithmetic` section for details
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
        accumulator = tl.dot(a, b, accumulator)
        # Advance the ptrs to the next K block.
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    # You can fuse arbitrary activation functions here
    # while the accumulator is still in FP32!
    if ACTIVATION == "leaky_relu":
        accumulator = leaky_relu(accumulator)
    c = accumulator.to(tl.float32)

    # -----------------------------------------------------------
    # Write back the block of the output matrix C with masks.
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, c, mask=c_mask)


# We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `matmul_kernel`.
@triton.jit
def leaky_relu(x):
    return tl.where(x >= 0, x, 0.01 * x)


# %%
# We can now create a convenience wrapper function that only takes two input tensors,
# and (1) checks any shape constraint; (2) allocates the output; (3) launches the above kernel.


def matmul_(a, b, c, activation=""):
    # Check constraints.
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    assert c.is_contiguous(), "Matrix C must be contiguous"
    M, K = a.shape
    K, N = b.shape
    assert c.shape == (M, N), "Output tensor has incorrect shape"
    # Allocates output.
    # c = torch.empty((M, N), device=a.device, dtype=torch.float32)
    # 1D launch kernel where each block gets its own program.
    grid = lambda META: (triton.cdiv(M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    matmul_kernel[grid](
        a, b, c,  #
        M, N, K,  #
        a.stride(0), a.stride(1),  #
        b.stride(0), b.stride(1),  #
        c.stride(0), c.stride(1),  #
        ACTIVATION=activation  #
    )
    return c


# # %%
# # Unit Test
# # ---------
# #
# # We can test our custom matrix multiplication operation against a native torch implementation (i.e., cuBLAS).

# torch.manual_seed(0)
# a = torch.randn((512, 512), device='cuda', dtype=torch.float32)
# b = torch.randn((512, 512), device='cuda', dtype=torch.float32)
# triton_output = matmul(a, b)
# torch_output = torch.matmul(a, b)
# print(f"triton_output_with_fp16_inputs={triton_output}")
# print(f"torch_output_with_fp16_inputs={torch_output}")
# # Bigger tolerance for AMD MI200 devices.
# # MI200 devices use reduced precision fp16 and bf16 and flush input and
# # output denormal values to zero. Detailed info is at: https://pytorch.org/docs/stable/notes/numerical_accuracy.html#reduced-precision-fp16-and-bf16-gemms-and-convolutions-on-amd-instinct-mi200-devices
# rtol = 1e-2 if is_hip_mi200() else 0
# if torch.allclose(triton_output, torch_output, atol=1e-2, rtol=rtol):
#     print("✅ Triton and Torch match")
# else:
#     print("❌ Triton and Torch differ")

# TORCH_HAS_FP8 = hasattr(torch, "float8_e5m2")
# if TORCH_HAS_FP8 and is_cuda():
#     torch.manual_seed(0)
#     a = torch.randn((512, 512), device="cuda", dtype=torch.float32)
#     b = torch.randn((512, 512), device="cuda", dtype=torch.float32)
#     a = a.to(torch.float8_e5m2)
#     # pre-transpose b for efficiency.
#     b = b.T
#     b = b.to(torch.float8_e5m2)
#     triton_output = matmul(a, b)
#     torch_output = torch.matmul(a.to(torch.float32), b.to(torch.float32))
#     print(f"triton_output_with_fp8_inputs={triton_output}")
#     print(f"torch_output_with_fp8_inputs={torch_output}")
#     if torch.allclose(triton_output, torch_output, atol=0.125, rtol=0):
#         print("✅ Triton and Torch match")
#     else:
#         print("❌ Triton and Torch differ")

# # %%
# # Benchmark
# # ---------
# #
# # Square Matrix Performance
# # ~~~~~~~~~~~~~~~~~~~~~~~~~~
# #
# # We can now compare the performance of our kernel against that of cuBLAS or rocBLAS. Here we focus on square matrices,
# # but feel free to arrange this script as you wish to benchmark any other matrix shape.

# ref_lib = 'cuBLAS' if is_cuda() else 'rocBLAS'

# configs = []
# for fp8_inputs in [False, True]:
#     if fp8_inputs and (not TORCH_HAS_FP8 or not is_cuda()):
#         continue
#     configs.append(
#         triton.testing.Benchmark(
#             x_names=["M", "N", "K"],  # Argument names to use as an x-axis for the plot
#             x_vals=[128 * i for i in range(2, 33)],  # Different possible values for `x_name`
#             line_arg="provider",  # Argument name whose value corresponds to a different line in the plot
#             # Possible values for `line_arg`
#             # Don't compare to cublas for fp8 cases as torch.matmul doesn't support fp8 at the moment.
#             line_vals=["triton"] if fp8_inputs else [ref_lib.lower(), "triton"],  # Label name for the lines
#             line_names=["Triton"] if fp8_inputs else [ref_lib, "Triton"],  # Line styles
#             styles=[("green", "-"), ("blue", "-")],
#             ylabel="TFLOPS",  # Label name for the y-axis
#             plot_name="matmul-performance-" +
#             ("fp16" if not fp8_inputs else "fp8"),  # Name for the plot, used also as a file name for saving the plot.
#             args={"fp8_inputs": fp8_inputs},
#         ))


# @triton.testing.perf_report(configs)
# def benchmark(M, N, K, provider, fp8_inputs):
#     a = torch.randn((M, K), device='cuda', dtype=torch.float32)
#     b = torch.randn((K, N), device='cuda', dtype=torch.float32)
#     if TORCH_HAS_FP8 and fp8_inputs:
#         a = a.to(torch.float8_e5m2)
#         b = b.T
#         b = b.to(torch.float8_e5m2)
#     quantiles = [0.5, 0.2, 0.8]
#     if provider == ref_lib.lower():
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: torch.matmul(a, b), quantiles=quantiles)
#     if provider == 'triton':
#         ms, min_ms, max_ms = triton.testing.do_bench(lambda: matmul(a, b), quantiles=quantiles)
#     perf = lambda ms: 2 * M * N * K * 1e-12 / (ms * 1e-3)
#     return perf(ms), perf(max_ms), perf(min_ms)


# benchmark.run(show_plots=True, print_data=True)