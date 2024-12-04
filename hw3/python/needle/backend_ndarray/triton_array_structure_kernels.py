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
def convert_index_to_location(
  index,
  offset,
  strides_ptr,
  shape_ptr,
  num_dims,
):
  final_index = offset
  current_size = 1
  prev_size = 1

  for i in range(MAX_SHAPE_DIMS - 1, -1, -1):
    if i < num_dims:
      shape_i = tl.load(shape_ptr + i, mask=True)

      current_size = prev_size * shape_i
      dim_index = (index % current_size) // prev_size
      
      strides_i = tl.load(strides_ptr + i, mask=True)

      final_index += dim_index * strides_i
      prev_size = current_size

  return final_index
  

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
      if OP_TYPE == OP_TYPE_COMPACT:
        val = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + i, val)

      elif OP_TYPE == OP_TYPE_EWISE_SET_ITEM:
        val = tl.load(input_ptr + i)
        tl.store(output_ptr + input_offset, val)

      elif OP_TYPE == OP_TYPE_SCALAR_SET_ITEM:
        tl.store(output_ptr + input_offset, scalar_val)

      else:
        assert False






