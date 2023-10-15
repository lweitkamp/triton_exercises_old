import torch

import triton
import triton.language as tl


###############################################################################
# Softmax full-row blocked                                                    #
###############################################################################


@triton.jit
def softmax_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    input_stride_x, input_stride_y,
    output_stride_x, output_stride_y,
):
    """Calculate the softmax of a row of the input tensor, storing the result
    in the output. We assume the input row fits into SRAM.

    Args:
        input_ptr: Pointer to the input tensor.
        output_ptr: Pointer to the output tensor.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        input_stride_x: Stride of the input tensor along the row dim.
        input_stride_y: Stride of the input tensor along the column dim.
        output_stride_x: Stride of the output tensor along the row dim.
        output_stride_y: Stride of the output tensor along the column dim.
    """
    program_id = tl.program_id(axis=0)

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_x, input_stride_y),
        offsets=(program_id, 0),
        block_shape=(1, N),
        order=(1, 0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N),
        strides=(output_stride_x, output_stride_y),
        offsets=(program_id, 0),
        block_shape=(1, N),
        order=(1, 0),
    )

    input_block = tl.load(
        input_block_ptr,
        boundary_check=(0, 1),
        padding_option="zero",
    )

    # Calculate softmax, don't forget the max trick.
    numerator = tl.exp(input_block - tl.max(input_block))
    denominator = tl.sum(numerator)
    softmax_output = numerator / denominator

    tl.store(output_block_ptr, softmax_output)


def softmax_blocked_row(inputs: torch.Tensor) -> torch.Tensor:
    """Calculate the softmax of a tensor along the final dim.

    Args:
        inputs: Tensor of shape (M, N) containing the input values.

    Returns:
        Tensor of shape (M, N) containing the softmax values.
    """
    M, N = inputs.shape
    outputs = torch.empty((M, N), dtype=inputs.dtype, device=inputs.device)

    softmax_kernel[(M, )](
        input_ptr=inputs, output_ptr=outputs,
        M=M, N=N,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
        output_stride_x=outputs.stride(0), output_stride_y=outputs.stride(1),
    )

    return outputs
