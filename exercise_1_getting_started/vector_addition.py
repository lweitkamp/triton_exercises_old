import torch

import triton
import triton.language as tl


###############################################################################
# Vector Addition (blocked)                                                   #
###############################################################################


@triton.jit
def vector_addition_kernel(
    a_ptr: tl.tensor,
    b_ptr: tl.tensor,
    output_ptr: tl.tensor,
    N: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    """Calculate the sum of two blocks of vectors.

    Args:
        a: Pointer to vector a.
        b: Pointer to vector b.
        output_ptr: Pointer to the output vector.
        M: Length of the input vector.
        BLOCK_M: Length of the block for which we calculate the sum of a and b.
    """
    program_id = tl.program_id(axis=0)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=(N, ),
        strides=(1, ),
        offsets=(program_id * BLOCK_N, ),
        block_shape=(BLOCK_N, ),
        order=(0, ),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=(N, ),
        strides=(1, ),
        offsets=(program_id * BLOCK_N, ),
        block_shape=(BLOCK_N, ),
        order=(0, ),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(N, ),
        strides=(1, ),
        offsets=(program_id * BLOCK_N, ),
        block_shape=(BLOCK_N, ),
        order=(0, ),
    )

    a_block = tl.load(a_block_ptr)
    b_block = tl.load(b_block_ptr)

    tl.store(output_block_ptr, a_block + b_block)


def vector_addition(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Calculate the sum of two vectors.

    Args:
        a: Vector of length N.
        b: Vector of length N.

    Returns:
        Vector of length N, the vector addition of a and b.
    """
    assert a.shape == b.shape
    assert a.ndim == 1

    N = a.shape[0]

    outputs = torch.empty((N, ), dtype=a.dtype, device=a.device)

    vector_addition_kernel[lambda meta: (triton.cdiv(N, meta['BLOCK_N']), )](
        a_ptr=a, b_ptr=b, output_ptr=outputs,
        N=N,
        BLOCK_N=16,
    )
    return outputs
