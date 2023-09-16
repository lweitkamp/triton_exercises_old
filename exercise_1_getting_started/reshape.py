import torch

import triton
import triton.language as tl


###############################################################################
# Reshape (naive) (unstable)                                                  #
###############################################################################


@triton.jit
def reshape_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
    input_stride_x, input_stride_y, input_stride_z,
    output_stride_x, output_stride_y,
):
    """Reshape the input tensor of size M by N by K into an output tensor of
    size M by (N * K). This kernel is naive and does not use any blocking,
    and will not scale up to tensors that exceed SRAM capability. Additionally,
    this kernel will not work at all since `tl.view` is not correctly
    implemented.

    Args:
        input_ptr: Pointer to the input tensor.
        output_ptr: Pointer to the output tensor.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        K: Number of channels in the input tensor.
        input_stride_x: Stride of the input tensor along the row dim.
        input_stride_y: Stride of the input tensor along the column dim.
        input_stride_z: Stride of the input tensor along the channel dim.
        output_stride_x: Stride of the output tensor along the row dim.
        output_stride_y: Stride of the output tensor along the column dim.
    """
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N, K),
        strides=(input_stride_x, input_stride_y, input_stride_z),
        offsets=(0, 0, 0),
        block_shape=(M, N, K),
        order=(2, 1, 0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(M, N * K),
        strides=(output_stride_x, output_stride_y),
        offsets=(0, 0),
        block_shape=(M, N * K),
        order=(1, 0),
    )

    input_block = tl.load(input_block_ptr)

    tl.store(output_block_ptr, tl.view(input_block, (M, N * K)))


def reshape(inputs: torch.Tensor) -> torch.Tensor:
    """Reshape a tensor using a naive reshape kernel.

    Args:
        inputs: Tensor of shape (M, N, K).

    Returns:
        Tensor of shape (M, N * K) - a reshaped version of the input tensor.
    """
    M, N, K = inputs.shape
    outputs = torch.empty((M, N * K), dtype=inputs.dtype, device=inputs.device)

    reshape_kernel[(1, )](
        input_ptr=inputs, output_ptr=outputs,
        M=M, N=N, K=K,
        input_stride_x=inputs.stride(0),
        input_stride_y=inputs.stride(1),
        input_stride_z=inputs.stride(2),
        output_stride_x=outputs.stride(0),
        output_stride_y=outputs.stride(1),
    )

    return outputs
