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
    return np.ascontiguousarray(strided_tensor.cpu().numpy())

def from_numpy(a, out):
    # Convert numpy array to tensor and copy to GPU
    tensor = torch.from_numpy(a.flatten()).cuda().contiguous()
    out.array.copy_(tensor)

def ewise_add(a, b, out):
    # add(a.array, b.array, out.array)
    ewise_op(a.array, b.array, out.array, operation="add")
def scalar_add(a, val, out):
    out.array[:] = a.array + val

def ewise_mul(a, b, out):
    # out.array[:] = a.array * b.array
    ewise_op(a.array, b.array, out.array, operation="multiply")

def scalar_mul(a, val, out):
    out.array[:] = a.array * val

def ewise_div(a, b, out):
    # out.array[:] = a.array / b.array
    ewise_op(a.array, b.array, out.array, operation="divide")

def scalar_div(a, val, out):
    out.array[:] = a.array / val

def ewise_sub(a, b, out):
    # out.array[:] = a.array - b.array
    ewise_op(a.array, b.array, out.array, operation="subtract")

def scalar_sub(a, val, out):
    out.array[:] = a.array - val

def ewise_eq(a, b, out):
    ewise_op(a.array, b.array, out.array, operation="equals")

def scalar_eq(a, val, out):
    out.array[:] = a.array == val

def ewise_ge(a, b, out):
    ewise_op(a.array, b.array, out.array, operation="greater_equal")

def scalar_ge(a, val, out):
    out.array[:] = a.array >= val




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
                     x - y)))))))))  # default case is subtraction
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

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
    }
    op_code = op_map.get(operation.lower(), 0)  # default to add if unknown operation
    
    # Launch kernel
    n_elements = output.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    ewise_op_kernel[grid](x, y, output, n_elements, op_code, BLOCK_SIZE=1024)  
    return output


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


# # From : https://isamu-website.medium.com/understanding-the-triton-tutorials-part-1-6191b59ba4c

# # `triton.jit`'ed functions can be auto-tuned by using the `triton.autotune` decorator, which consumes:
# #   - A list of `triton.Config` objects that define different configurations of
# #       meta-parameters (e.g., `BLOCK_SIZE_M`) and compilation options (e.g., `num_warps`) to try
# #   - An auto-tuning *key* whose change in values will trigger evaluation of all the
# #       provided configs
# @triton.autotune(
#     configs=[
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4, num_warps=4),
#         triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#         triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5, num_warps=2),
#     ],
#     key=['M', 'N', 'K'],
# )
# @triton.jit
# def matmul_kernel(
#     # Pointers to matrices
#     a_ptr, b_ptr, c_ptr,
#     # Matrix dimensions
#     M, N, K,
#     # The stride variables represent how much to increase the ptr by when moving by 1
#     # element in a particular dimension. E.g. `stride_am` is how much to increase `a_ptr`
#     # by to get the element one row down (A has M rows).
#     stride_am, stride_ak,
#     stride_bk, stride_bn,
#     stride_cm, stride_cn,
#     # Meta-parameters
#     BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
#     GROUP_SIZE_M: tl.constexpr,
#     ACTIVATION: tl.constexpr,
# ):
#     """Kernel for computing the matmul C = A x B.
#     A has shape (M, K), B has shape (K, N) and C has shape (M, N)
#     """
#     # -----------------------------------------------------------
#     # Map program ids `pid` to the block of C it should compute.
#     # This is done in a grouped ordering to promote L2 data reuse.
#     # See above `L2 Cache Optimizations` section for details.
#     pid = tl.program_id(axis=0)
#     num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
#     num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
#     num_pid_in_group = GROUP_SIZE_M * num_pid_n
#     group_id = pid // num_pid_in_group
#     first_pid_m = group_id * GROUP_SIZE_M
#     group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
#     pid_m = first_pid_m + (pid % group_size_m)
#     pid_n = (pid % num_pid_in_group) // group_size_m

#     # ----------------------------------------------------------
#     # Create pointers for the first blocks of A and B.
#     # We will advance this pointer as we move in the K direction
#     # and accumulate
#     # `a_ptrs` is a block of [BLOCK_SIZE_M, BLOCK_SIZE_K] pointers
#     # `b_ptrs` is a block of [BLOCK_SIZE_K, BLOCK_SIZE_N] pointers
#     # See above `Pointer Arithmetics` section for details
#     offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
#     offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
#     offs_k = tl.arange(0, BLOCK_SIZE_K)
#     a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
#     b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

#     # -----------------------------------------------------------
#     # Iterate to compute a block of the C matrix.
#     # We accumulate into a `[BLOCK_SIZE_M, BLOCK_SIZE_N]` block
#     # of fp32 values for higher accuracy.
#     # `accumulator` will be converted back to fp16 after the loop.
#     accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
#     for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
#         # Load the next block of A and B, generate a mask by checking the K dimension.
#         # If it is out of bounds, set it to 0.
#         a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
#         b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
#         # We accumulate along the K dimension.
#         accumulator += tl.dot(a, b)
#         # Advance the ptrs to the next K block.
#         a_ptrs += BLOCK_SIZE_K * stride_ak
#         b_ptrs += BLOCK_SIZE_K * stride_bk
#     # You can fuse arbitrary activation functions here
#     # while the accumulator is still in FP32!
#     if ACTIVATION == "leaky_relu":
#         accumulator = leaky_relu(accumulator)
#     c = accumulator.to(tl.float16)

#     # -----------------------------------------------------------
#     # Write back the block of the output matrix C with masks.
#     offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
#     offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
#     c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
#     c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
#     tl.store(c_ptrs, c, mask=c_mask)


# # We can fuse `leaky_relu` by providing it as an `ACTIVATION` meta-parameter in `_matmul`.
# @triton.jit
# def leaky_relu(x):
#     x = x + 1
#     return tl.where(x >= 0, x, 0.01 * x)

# def matmul(a, b, out, m, n, p):
#     # Check constraints.
#     # assert a.shape[1] == b.shape[0], "Incompatible dimensions"
#     # assert a.is_contiguous(), "Matrix A must be contiguous"
#     # assert b.is_contiguous(), "Matrix B must be contiguous"
#     # m, n = a.shape
#     # n, p = b.shape
#     # Allocates output.
#     c = torch.empty((m, p), device=a.device, dtype=a.dtype)
#     # 1D launch kernel where each block gets its own program.
#     grid = lambda META: (
#         triton.cdiv(m, META['BLOCK_SIZE_M']) * triton.cdiv(p, META['BLOCK_SIZE_N']),
#     )
#     matmul_kernel[grid](
#         a, b, out,
#         m, p, n,
#         a.stride(0), a.stride(1),
#         b.stride(0), b.stride(1),
#         c.stride(0), c.stride(1),
#         ACTIVATION=""
#     )
#     return c

