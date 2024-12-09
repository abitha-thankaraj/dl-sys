import triton
import triton.language as tl


__default_data_type__ = tl.float32


OP_TYPE_REDUCE_MAX: tl.constexpr = 1
OP_TYPE_REDUCE_SUM: tl.constexpr = 2


REDUCTION_OPERATION_MAP = {
  "REDUCE_MAX": OP_TYPE_REDUCE_MAX,
  "REDUCE_SUM": OP_TYPE_REDUCE_SUM,
}


@triton.jit
def reduction_kernel(
  input_ptr,
  output_ptr,
  n_rows,
  reduce_size,
  OP_TYPE,
  BLOCK_SIZE: tl.constexpr,
  REDUCE_SIZE_PER_BLOCK: tl.constexpr,
):
  """
  Divides the data in the input array into buckets of size
  reduce_size, calculates the maximum/summation of those elements
  (depending on given OP_TYPE), and stores these elements in
  the output array.

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

    n_rows:
      Number of rows for which we have to perform the operation
      
    reduce_size:
      The size of each bucket that we need to perform the reduction
      on

    OP_TYPE:
      operation type, used to switch between maximum/summation

    REDUCE_SIZE_PER_BLOCK:
      Number of elements to perform reduction on using a given
      GPU Block
  """
  # Get program ID
  assert OP_TYPE == OP_TYPE_REDUCE_MAX or OP_TYPE == OP_TYPE_REDUCE_SUM
  pid = tl.program_id(0)
    
  # Calculate row this thread block is processing
  row_idx = pid

  # Cast the custom pointers from CudaArray class
  # to a format that triton can read from
  output_ptr = tl.cast(
    output_ptr, 
    tl.pointer_type(__default_data_type__),
  )

  input_ptr = tl.cast(
    input_ptr, 
    tl.pointer_type(__default_data_type__),
  )
    
  # Only process if row is within bounds
  if row_idx < n_rows:
    # Initialize max value with first element
    base_idx = row_idx * reduce_size

    reduced_val = tl.load(input_ptr + base_idx)
        
    # Process elements in chunks within the reduction dimension
    for i in range(1, reduce_size, REDUCE_SIZE_PER_BLOCK):
      # Calculate element index
      elem_idx = base_idx + i
            
      # Process elements in this chunk
      for j in range(REDUCE_SIZE_PER_BLOCK):
        if i + j < reduce_size:
          val = tl.load(input_ptr + elem_idx + j)

          if OP_TYPE == OP_TYPE_REDUCE_MAX:
            reduced_val = tl.maximum(reduced_val, val)
          
          else:
            reduced_val = reduced_val + val
        
    # Store result
    tl.store(output_ptr + row_idx, reduced_val)