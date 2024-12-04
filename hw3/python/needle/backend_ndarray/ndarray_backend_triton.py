import triton
import numpy as np


from .triton_array_structure_kernels import (
  fill_kernel,
  modify_structure_kernel,
  STRUCTURE_TRITON_OPERATION_MAP,
)

from .triton_ewise_or_scalar_ops_kernels import (
  OPERATION_MAP,
  elementwise_binary_op_kernel,
  scalar_binary_op_kernel,
  unary_op_kernel,
)

from .triton_reduction_kernels import (
  reduction_kernel,
  REDUCTION_OPERATION_MAP,
)

from .triton_matmul_kernel import (
  matmul_kernel,
)

from .triton_array import *


BLOCK_SIZE = 256
REDUCE_SIZE_PER_BLOCK = 32
TILE = 16


def fill(out, val):
  grid = (triton.cdiv(out.size, BLOCK_SIZE),)
  fill_kernel[grid](
    out_ptr=out.ptr(),
    val=val,
    n_elements=out.size,
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_add(a, b, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  elementwise_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_b=b.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["ADDITION"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_mul(a, b, out):    
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  elementwise_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_b=b.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["MULTIPLICATION"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_div(a, b, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  elementwise_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_b=b.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["DIVISION"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_maximum(a, b, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  elementwise_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_b=b.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["MAXIMUM"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_eq(a, b, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  elementwise_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_b=b.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["EQ"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_ge(a, b, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  elementwise_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_b=b.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["GE"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def scalar_add(a, val, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  scalar_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    val=val,
    n_elements=out.size,
    op=OPERATION_MAP["ADDITION"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def scalar_mul(a, val, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  scalar_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    val=val,
    n_elements=out.size,
    op=OPERATION_MAP["MULTIPLICATION"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def scalar_div(a, val, out):
  if val == 0:
    raise Exception("Division by zero not allowed!")

  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  scalar_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    val=val,
    n_elements=out.size,
    op=OPERATION_MAP["DIVISION"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def scalar_power(a, val, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  scalar_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    val=val,
    n_elements=out.size,
    op=OPERATION_MAP["POWER"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def scalar_maximum(a, val, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  scalar_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    val=val,
    n_elements=out.size,
    op=OPERATION_MAP["MAXIMUM"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def scalar_ge(a, val, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  scalar_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    val=val,
    n_elements=out.size,
    op=OPERATION_MAP["GE"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def scalar_eq(a, val, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  scalar_binary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    val=val,
    n_elements=out.size,
    op=OPERATION_MAP["EQ"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_exp(a, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  unary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["EXP"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_log(a, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  unary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["LOG"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_tanh(a, out):
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  unary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["TANH"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def compact(a, out, shape, strides, offset):
  # create and host the shape array
  shapeArray = ShapeArray(len(shape))
  shape = np.ascontiguousarray(np.array(shape, dtype=np.int32))
  shape_array_from_numpy(shape, shapeArray)

  # create and host the strides array
  strideArray = ShapeArray(len(strides))
  strides = np.ascontiguousarray(np.array(strides, dtype=np.int32))
  shape_array_from_numpy(strides, strideArray)

  # Run Kernel
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  modify_structure_kernel[grid](
    input_ptr=a.ptr(),
    output_ptr=out.ptr(),
    shape_ptr=shapeArray.ptr(),
    stride_ptr=strideArray.ptr(),
    n_elements=out.size,
    n_dims=len(shape),
    offset=int(offset),
    scalar_val=-1,
    OP_TYPE=STRUCTURE_TRITON_OPERATION_MAP["COMPACT"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_setitem(a, out, shape, strides, offset):
  # create and host the shape array
  shapeArray = ShapeArray(len(shape))
  shape = np.ascontiguousarray(np.array(shape, dtype=np.int32))
  shape_array_from_numpy(shape, shapeArray)

  # create and host the strides array
  strideArray = ShapeArray(len(strides))
  strides = np.ascontiguousarray(np.array(strides, dtype=np.int32))
  shape_array_from_numpy(strides, strideArray)

  # Calculate total number of elements
  n_elements = 1
  for s in shape:
    n_elements *= s

  n_elements = int(n_elements)

  # Run Kernel
  grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
  modify_structure_kernel[grid](
    input_ptr=a.ptr(),
    output_ptr=out.ptr(),
    shape_ptr=shapeArray.ptr(),
    stride_ptr=strideArray.ptr(),
    n_elements=n_elements,
    n_dims=len(shape),
    offset=int(offset),
    scalar_val=-1,
    OP_TYPE=STRUCTURE_TRITON_OPERATION_MAP["EWISE_SET_ITEM"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def scalar_setitem(size, val, out, shape, strides, offset):
  # create and host the shape array
  shapeArray = ShapeArray(len(shape))
  shape = np.ascontiguousarray(np.array(shape, dtype=np.int32))
  shape_array_from_numpy(shape, shapeArray)

  # create and host the strides array
  strideArray = ShapeArray(len(strides))
  strides = np.ascontiguousarray(np.array(strides, dtype=np.int32))
  shape_array_from_numpy(strides, strideArray)

  # Calculate total number of elements
  n_elements = 1
  for s in shape:
    n_elements *= s

  n_elements = int(n_elements)

  # Run Kernel
  grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
  modify_structure_kernel[grid](
    input_ptr=out.ptr(), # Placeholder, not used for this code
    output_ptr=out.ptr(),
    shape_ptr=shapeArray.ptr(),
    stride_ptr=strideArray.ptr(),
    n_elements=n_elements,
    n_dims=len(shape),
    offset=int(offset),
    scalar_val=val,
    OP_TYPE=STRUCTURE_TRITON_OPERATION_MAP["SCALAR_SET_ITEM"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def reduce_max(a, out, reduce_size):
  n_rows = a.size // reduce_size
  assert n_rows == out.size
    
  # Calculate grid size - one thread block per row
  grid = (n_rows,)
    
  # Launch kernel
  reduction_kernel[grid](
    input_ptr=a.ptr(),
    output_ptr=out.ptr(),
    n_rows=int(n_rows),
    reduce_size=int(reduce_size),
    OP_TYPE=REDUCTION_OPERATION_MAP["REDUCE_MAX"],
    BLOCK_SIZE=BLOCK_SIZE,
    REDUCE_SIZE_PER_BLOCK=REDUCE_SIZE_PER_BLOCK,
  )


def reduce_sum(a, out, reduce_size):
  n_rows = a.size // reduce_size
  assert n_rows == out.size
    
  # Calculate grid size - one thread block per row
  grid = (n_rows,)

  print(REDUCTION_OPERATION_MAP["REDUCE_SUM"])
    
  # Launch kernel
  reduction_kernel[grid](
    input_ptr=a.ptr(),
    output_ptr=out.ptr(),
    n_rows=int(n_rows),
    reduce_size=int(reduce_size),
    OP_TYPE=REDUCTION_OPERATION_MAP["REDUCE_SUM"],
    BLOCK_SIZE=BLOCK_SIZE,
    REDUCE_SIZE_PER_BLOCK=REDUCE_SIZE_PER_BLOCK,
  )


def matmul(a, b, out, m, n, p):
  # Compute grid
  grid = (
    (m + TILE - 1) // TILE,
    (p + TILE - 1) // TILE,
  )

  # We define strides assuming a row-major layout in this code
  # Launch Kernel
  matmul_kernel[grid](
    a_ptr=a.ptr(),
    b_ptr=b.ptr(), 
    c_ptr=out.ptr(),
    M=m, 
    N=n, 
    P=p,
    stride_am=n, 
    stride_an=1,
    stride_bn=p, 
    stride_bp=1,
    stride_cm=p, 
    stride_cp=1,
    BLOCK_SIZE=TILE,
  )




