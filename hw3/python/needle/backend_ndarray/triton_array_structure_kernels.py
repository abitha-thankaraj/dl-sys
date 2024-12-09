import triton
import triton.language as tl
from typing import Dict


__default_data_type__ = tl.float32

MAX_SHAPE_DIMS: tl.constexpr = 8

# Operation types defined here
OP_TYPE_COMPACT: tl.constexpr = 1
OP_TYPE_EWISE_SET_ITEM: tl.constexpr = 2
OP_TYPE_SCALAR_SET_ITEM: tl.constexpr = 3


STRUCTURE_TRITON_OPERATION_MAP: Dict[str, tl.constexpr] = {
  "COMPACT": OP_TYPE_COMPACT,
  "EWISE_SET_ITEM": OP_TYPE_EWISE_SET_ITEM,
  "SCALAR_SET_ITEM": OP_TYPE_SCALAR_SET_ITEM,
}


@triton.jit
def fill_kernel(
  out_ptr,  
  val,     
  n_elements,     
  BLOCK_SIZE: tl.constexpr,
):
  """
  Given a ptr to the array, and number of elements,
  this kernel populates the array with the given scalar value.

  Input:
    out_ptr:
      pointer to the output memory

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    val:
      scalar value that the array will be populated with

    n_elements:
      Number of elements in the array

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
  out_ptr = tl.cast(
    out_ptr, 
    tl.pointer_type(__default_data_type__),
  )

  # Fill the array with the specified value where mask is True
  tl.store(out_ptr + offs, val, mask=mask)
  

@triton.jit
def modify_structure_kernel(
  input_ptr,
  output_ptr,
  shape_ptr,
  stride_ptr,
  n_elements,
  n_dims,
  offset,
  scalar_val,
  OP_TYPE: tl.constexpr,
  BLOCK_SIZE: tl.constexpr,
):
  """
  This kernel performs 3 operations (compressed into one Kernel 
  to reduce code duplication since a lot of the operations are common
  between them):
    1. compact: copy the elements of the array into another array
                making it contiguous/compact in memory
    2. ewise_setitem: copy the elements of one array into another array
                      respecting the shape/strides/offsets given
    3. scalar_setitem: use a scalar to populate the elements in the
                       output array, respecting the shape/strides/offsets given

  Input:
    input_ptr:
      pointer to the input memory

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    output_ptr:
      pointer to the output memory

      NOTE: this is using our C++ memory implementation, so
      needs to cast in tl.float32 in order for Triton to be
      able to read the memory

    shape_ptr:
      pointer to an array, holding the shape informating of the array

      NOTE: this is using our C++ memory implementation, so needs
      to be cast to tl.int32 for triton to be able to read the memory

    stride_ptr:
      pointer to an array, holding the stride informating of the array

      NOTE: this is using our C++ memory implementation, so needs
      to be cast to tl.int32 for triton to be able to read the memory

    n_elements:
      Number of elements in the array

    n_dims:
      Number of dimensions, which is also the length of the shape/stride
      array

    offset:
      Offset from which we have to start the operation 

    scalar_val:
      scalar value that will be used to populate the array

      NOTE: this is only used when we are using this kernel for 
      scalar_setitem

    OP_TYPE:
      operation type, used to switch between compact / ewise_setitem / scalar_setitem

    BLOCK_SIZE:
      size of the GPU block/number of threads
  """
  # Get program ID
  pid = tl.program_id(0)
    
  # Compute the block start index
  block_start = pid * BLOCK_SIZE

  output_ptr = tl.cast(
    output_ptr, 
    tl.pointer_type(__default_data_type__),
  )
  input_ptr = tl.cast(
    input_ptr, 
    tl.pointer_type(__default_data_type__),
  )
  shape_ptr = tl.cast(
    shape_ptr, 
    tl.pointer_type(tl.int32),
  )
  stride_ptr = tl.cast(
    stride_ptr,
    tl.pointer_type(tl.int32),
  )
    
  # Process elements in the block
  for i in range(block_start, block_start + BLOCK_SIZE):
    if i < n_elements:
      # Initialize input offset with base offset
      input_offset = offset
            
      current_size = 1
      previous_size = 1

      for dim in range(n_dims - 1, -1, -1):
        curr_shape = tl.load(shape_ptr + dim)
        curr_stride = tl.load(stride_ptr + dim)

        current_size = previous_size * curr_shape
        indices_i = (i % current_size) // previous_size

        input_offset += indices_i * curr_stride
        previous_size = current_size
            
      # Load and store value
      # Compact operation
      if OP_TYPE == OP_TYPE_COMPACT:
        val = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + i, val)

      # Ewise setitem operation
      elif OP_TYPE == OP_TYPE_EWISE_SET_ITEM:
        val = tl.load(input_ptr + i)
        tl.store(output_ptr + input_offset, val)

      # Scalar setitem operation
      elif OP_TYPE == OP_TYPE_SCALAR_SET_ITEM:
        tl.store(output_ptr + input_offset, scalar_val)

      else:
        assert False