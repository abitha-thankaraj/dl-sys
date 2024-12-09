# NDArray backend implementation using Triton Kernels

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
  """
  Given a pointer to the output and a scalar value,
  fills all the elements of the output array
  with the given scalar value.
  """
  grid = (triton.cdiv(out.size, BLOCK_SIZE),)
  fill_kernel[grid](
    out_ptr=out.ptr(),
    val=val,
    n_elements=out.size,
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_add(a, b, out):
  """
  Elementwise addition
  """
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
  """
  Elementwise multiplication
  """  
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
  """
  Elementwise division
  """
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
  """
  Elementwise maximum
  out[i] = max(a[i], b[i]) for all i
  """
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
  """
  Elementwise equality
  out[i] = (a[i] == b[i]) for all i
  """
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
  """
  Elementwise greater than or equal
  out[i] = (a[i] >= b[i]) for all i
  """
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
  """
  Adds an scalar to all elements in a given array
  """
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
  """
  Multiplies a scalar with all elements in a given array
  """
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
  """
  Divides all elements of an array with a given scalar
  """
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
  """
  out[i] = a[i]^val for all i
  """
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
  """
  out[i] = max(a[i], val) for all i
  """
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
  """
  out[i] = (a[i] >= val) for all i
  """
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
  """
  out[i] = (a[i] == val) for all i
  """
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
  """
  out[i] = exp(a[i]) for all i
  """
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  unary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["EXP"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_log(a, out):
  """
  out[i] = log(a[i]) for all i
  """
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  unary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["LOG"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def ewise_tanh(a, out):
  """
  out[i] = tanh(a[i]) for all i
  """
  grid = (triton.cdiv(a.size, BLOCK_SIZE),)
  unary_op_kernel[grid](
    ptr_a=a.ptr(),
    ptr_out=out.ptr(),
    n_elements=out.size,
    op=OPERATION_MAP["TANH"],
    BLOCK_SIZE=BLOCK_SIZE,
  )


def compact(a, out, shape, strides, offset):
  """
  out = compact version of a, i.e., the memory is in
        contiguous form

  NOTE: Shape/stride can have arbitrary length,
        And Triton does not support passing tuples as argument
        due to the requirements of jit
  """
  # create and host the shape array
  # This is done so that triton can read the shape objects
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
  """
  Elementwise set elements in out array, respecting
  the shape/strides/offset, copy these elements from a

  NOTE: Shape/stride can have arbitrary length,
        And Triton does not support passing tuples as argument
        due to the requirements of jit
  """
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
  """
  Elementwise set elements in out array, respecting
  the shape/strides/offset, copied elements should be equal to
  the scalar val

  NOTE: Shape/stride can have arbitrary length,
        And Triton does not support passing tuples as argument
        due to the requirements of jit
  """
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
  """
  Takes each (reduce_size) number of elements of a,
  calculates the maximum, and sets the corresponding element in out.

  I.e.: out[i] = maximum(a[i * reduce_size : (i + 1) * reduce_size])
  """
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
    REDUCE_SIZE_PER_BLOCK=REDUCE_SIZE_PER_BLOCK,
  )


def reduce_sum(a, out, reduce_size):
  """
  Takes each (reduce_size) number of elements of a,
  calculates the summation, and sets the corresponding element in out.

  I.e.: out[i] = sum(a[i * reduce_size : (i + 1) * reduce_size])
  """
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
    REDUCE_SIZE_PER_BLOCK=REDUCE_SIZE_PER_BLOCK,
  )


def matmul(a, b, out, m, n, p):
  """
  Performs matrix multiplication of 2D arrays
  
  Inputs:
    a: first matrix, of shape m x n
    b: second matrix, of shape n x p
    out: output matrix, of shape m x p
    m: first dimension size of a
    n: second dimension size of a
    p: second dimension size of b
  """
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




