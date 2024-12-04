import triton
import triton.language as tl
from typing import Dict


__default_data_type__ = tl.float32


@triton.jit
def matmul_kernel(
    a_ptr, 
    b_ptr, 
    c_ptr,
    M, 
    N, 
    P,
    stride_am, 
    stride_an,
    stride_bn, 
    stride_bp,
    stride_cm, 
    stride_cp,
    BLOCK_SIZE: tl.constexpr,
):
    # Program IDs
    pid_m = tl.program_id(0)  # Rows
    pid_n = tl.program_id(1)  # Columns

    # Offsets for the current block
    block_row = pid_m * BLOCK_SIZE
    block_col = pid_n * BLOCK_SIZE

    # Create ranges for rows and columns within the block
    idx_m = block_row + tl.arange(0, BLOCK_SIZE)
    idx_n = block_col + tl.arange(0, BLOCK_SIZE)

    # Initialize the accumulator to zero (float32 for accumulation)
    acc = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    # Cast the custom pointers from CudaArray class
    # to a format that triton can read from
    a_ptr = tl.cast(
      a_ptr, 
      tl.pointer_type(__default_data_type__),
    )

    b_ptr = tl.cast(
      b_ptr, 
      tl.pointer_type(__default_data_type__),
    )

    c_ptr = tl.cast(
      c_ptr,
      tl.pointer_type(__default_data_type__),
    )

    # Pointer arithmetic
    a_ptrs = a_ptr + (idx_m[:, None] * stride_am + tl.arange(0, BLOCK_SIZE)[None, :] * stride_an)
    b_ptrs = b_ptr + (tl.arange(0, BLOCK_SIZE)[:, None] * stride_bn + idx_n[None, :] * stride_bp)

    # Loop over K dimension
    for k in range(0, N, BLOCK_SIZE):
        # Masks to handle out-of-bounds accesses
        a_mask = (idx_m[:, None] < M) & ((k + tl.arange(0, BLOCK_SIZE))[None, :] < N)
        b_mask = ((k + tl.arange(0, BLOCK_SIZE))[:, None] < N) & (idx_n[None, :] < P)

        # Load A and B tiles as float16
        a_tile = tl.load(a_ptrs + k * stride_an, mask=a_mask, other=0.0).to(tl.float16)
        b_tile = tl.load(b_ptrs + k * stride_bn, mask=b_mask, other=0.0).to(tl.float16)

        # Perform matrix multiplication on the tiles
        acc += tl.dot(a_tile, b_tile)

    # Write back the result to C
    c_mask = (idx_m[:, None] < M) & (idx_n[None, :] < P)
    c_ptrs = c_ptr + idx_m[:, None] * stride_cm + idx_n[None, :] * stride_cp
    
    # Cast the accumulator to the desired output type (e.g., float32)
    tl.store(c_ptrs, acc, mask=c_mask)
