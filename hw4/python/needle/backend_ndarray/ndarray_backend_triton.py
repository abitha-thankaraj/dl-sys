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
# def matmul(a, b, out, m, n, p):
#     matmul_op(a.array, b.array, out.array, m, n, p)


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
