import triton
import triton.language as tl
from typing import Dict


__default_data_type__ = tl.float32


OP_TYPE_ADDITION: tl.constexpr = 1
OP_TYPE_MULTIPLICATION: tl.constexpr = 2
OP_TYPE_DIVISION: tl.constexpr = 3
OP_TYPE_MAXIMUM: tl.constexpr = 4
OP_TYPE_EQ: tl.constexpr = 5
OP_TYPE_GE: tl.constexpr = 6
OP_TYPE_POWER: tl.constexpr = 7
OP_TYPE_LOG: tl.constexpr = 8
OP_TYPE_EXP: tl.constexpr = 9
OP_TYPE_TANH: tl.constexpr = 10


OPERATION_MAP: Dict[str, tl.constexpr] = {
  "ADDITION": OP_TYPE_ADDITION,
  "MULTIPLICATION": OP_TYPE_MULTIPLICATION,
  "DIVISION": OP_TYPE_DIVISION,
  "MAXIMUM": OP_TYPE_MAXIMUM,
  "EQ": OP_TYPE_EQ,
  "GE": OP_TYPE_GE,
  "POWER": OP_TYPE_POWER,
  "LOG": OP_TYPE_LOG,
  "EXP": OP_TYPE_EXP,
  "TANH": OP_TYPE_TANH,
}


@triton.jit
def elementwise_binary_op_kernel(
  ptr_a,
  ptr_b,
  ptr_out,
  n_elements: int,
  op: tl.constexpr,
  BLOCK_SIZE: tl.constexpr,
):
  """
  Performs elementwise operation between two
  array, a and b, and stores the result in output array.

  Input:
    ptr_a:
      Pointer to array a.

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    ptr_b:
      Pointer to array b

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    ptr_out:
      Pointer to the output array

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    n_elements:
      Number of elements in the array

    op:
      operation type, used to switch between different types
      of binary operation (eg., addition, multiplication etc.)

    BLOCK_SIZE:
      size of the GPU block/number of threads
  """
  # Calculate the absolute position
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE

  # create a mask for valid indices
  offs = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offs < n_elements

  # Cast the custom pointers from CudaArray class
  # to a format that triton can read from
  ptr_a = tl.cast(
    ptr_a, 
    tl.pointer_type(__default_data_type__),
  )
  ptr_b = tl.cast(
    ptr_b, 
    tl.pointer_type(__default_data_type__),
  )
  ptr_out = tl.cast(
    ptr_out,
    tl.pointer_type(__default_data_type__),
  )

  # load the data from the pointers
  a = tl.load(
    ptr_a + offs,
    mask=mask,
  )
  b = tl.load(
    ptr_b + offs,
    mask=mask,
  )

  # perform the operation
  if op == OP_TYPE_ADDITION:
    output = a + b

  elif op == OP_TYPE_MULTIPLICATION:
    output = a * b

  elif op == OP_TYPE_DIVISION:
    output = a / b

  elif op == OP_TYPE_MAXIMUM:
    output = tl.maximum(a, b)

  elif op == OP_TYPE_GE:
    output = (a >= b)

  elif op == OP_TYPE_EQ:
    output = (a == b)

  else:
    raise ValueError(f"Given operation type {op} is not supported.")

  # store the output in the output array
  tl.store(
    ptr_out + offs, 
    output, 
    mask=mask,
  )


@triton.jit
def scalar_binary_op_kernel(
  ptr_a,
  ptr_out,
  val,
  n_elements: int,
  op: tl.constexpr,
  BLOCK_SIZE: tl.constexpr,
):
  """
  Performs elementwise operation between elements of an array, a,
  and a scalar value, val, and stores the result in output array.

  Input:
    ptr_a:
      Pointer to array a.

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    ptr_out:
      Pointer to the output array

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    val:
      Scalar value that is to be used for the intended operation

    n_elements:
      Number of elements in the array

    op:
      operation type, used to switch between different types
      of binary operation (eg., addition, multiplication etc.)

    BLOCK_SIZE:
      size of the GPU block/number of threads
  """
  # Calculate the absolute position
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE

  # create a mask for valid indices
  offs = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offs < n_elements

  # Cast the custom pointers from CudaArray class
  # to a format that triton can read from
  ptr_a = tl.cast(
    ptr_a, 
    tl.pointer_type(__default_data_type__),
  )

  ptr_out = tl.cast(
    ptr_out,
    tl.pointer_type(__default_data_type__),
  )

  # load the data from the pointers
  a = tl.load(
    ptr_a + offs,
    mask=mask,
  )

  # perform the operation
  if op == OP_TYPE_ADDITION:
    output = a + val

  elif op == OP_TYPE_MULTIPLICATION:
    output = a * val

  elif op == OP_TYPE_DIVISION:
    output = a / val

  elif op == OP_TYPE_POWER:
    # Initialize result with x (will be modified based on conditions)
    output = a
    
    # Handle integer powers specially for better precision
    is_int_power = (val == tl.floor(val))
    
    # Case 1: Integer powers
    if is_int_power:
        # Handle negative x with even/odd powers
        is_even = tl.floor(val) % 2 == 0
        sign = tl.where(a < 0, -1.0, 1.0)
        abs_a = tl.abs(a)
        
        # For even powers, result is always positive
        # For odd powers, preserve sign
        power_result = tl.exp(tl.log(abs_a + 1e-30) * val)
        output = tl.where(is_even, power_result, power_result * sign)
        
        # Special case for x = 0
        output = tl.where(a == 0, 0.0, output)
        
    # Case 2: Non-integer powers
    else:
        # For non-integer powers, x must be non-negative
        is_valid = (a >= 0)
        power_result = tl.exp(tl.log(a + 1e-30) * val)
        output = tl.where(is_valid, power_result, float('nan'))
        
        # Special case for x = 0
        output = tl.where(a == 0, 0.0, output)

  elif op == OP_TYPE_MAXIMUM:
    output = tl.maximum(a, val)

  elif op == OP_TYPE_GE:
    output = (a >= val)

  elif op == OP_TYPE_EQ:
    output = (a == val)

  else:
    assert False

  # store the output in the output array
  tl.store(
    ptr_out + offs, 
    output, 
    mask=mask,
  )


@triton.jit
def unary_op_kernel(
  ptr_a,
  ptr_out,
  n_elements: int,
  op: tl.constexpr,
  BLOCK_SIZE: tl.constexpr,
):
  """
  Performs elementwise unary operations on an array, a,
  and stores the result in output array.

  Input:
    ptr_a:
      Pointer to array a.

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    ptr_out:
      Pointer to the output array

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    n_elements:
      Number of elements in the array

    op:
      operation type, used to switch between different types
      of binary operation (eg., exp, log etc.)

    BLOCK_SIZE:
      size of the GPU block/number of threads
  """
  # Calculate the absolute position
  pid = tl.program_id(axis=0)
  block_start = pid * BLOCK_SIZE

  # create a mask for valid indices
  offs = block_start + tl.arange(0, BLOCK_SIZE)
  mask = offs < n_elements

  # Cast the custom pointers from CudaArray class
  # to a format that triton can read from
  ptr_a = tl.cast(
    ptr_a, 
    tl.pointer_type(__default_data_type__),
  )

  ptr_out = tl.cast(
    ptr_out,
    tl.pointer_type(__default_data_type__),
  )

  # load the data from the pointers
  a = tl.load(
    ptr_a + offs,
    mask=mask,
  )

  # perform the operation
  if op == OP_TYPE_EXP:
    output = tl.exp(a)

  elif op == OP_TYPE_LOG:
    output = tl.log(a)

  elif op == OP_TYPE_TANH:
    output = (tl.exp(a) - tl.exp(-a)) / (tl.exp(a) + tl.exp(-a))

  else:
    assert False

  # store the output in the output array
  tl.store(
    ptr_out + offs, 
    output, 
    mask=mask,
  )