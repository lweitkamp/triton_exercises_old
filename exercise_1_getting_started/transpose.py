import torch

import triton
import triton.language as tl


###############################################################################
# Transpose kernel (naive)                                                    #
###############################################################################


@triton.jit
def transpose_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    input_stride_x, input_stride_y,
    output_stride_x, output_stride_y,
):
    """Transpose a row of the input tensor, storing the result in the output.

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
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(M, N),
        strides=(input_stride_x, input_stride_y),
        offsets=(0, 0),
        block_shape=(M, N),
        order=(1, 0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(N, M),
        strides=(output_stride_x, output_stride_y),
        offsets=(0, 0),
        block_shape=(N, M),
        order=(1, 0),
    )

    input_block = tl.load(input_block_ptr)

    tl.store(output_block_ptr, tl.trans(input_block))


def transpose(inputs: torch.Tensor) -> torch.Tensor:
    """Transpose a tensor of shape (M, N) to (N, M). This kernel is naive and
    does not use any blocking, and will not scale up to tensors that exceed
    SRAM capability.

    Args:
        inputs: Tensor of shape (M, N).

    Returns:
        Tensor of shape (N, M) - a transpose of the input tensor.
    """
    M, N = inputs.shape
    outputs = torch.empty((N, M), dtype=inputs.dtype, device=inputs.device)

    transpose_kernel[(1, )](
        input_ptr=inputs, output_ptr=outputs,
        M=M, N=N,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
        output_stride_x=outputs.stride(0), output_stride_y=outputs.stride(1),
    )

    return outputs


###############################################################################
# Transpose kernel pointer trick (naive)                                      #
###############################################################################


@triton.jit
def transpose_with_block_ptr_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    input_stride_x, input_stride_y,
    output_stride_x, output_stride_y,
):
    """Transpose a tensor of shape (M, N) to (N, M) by loading the input tensor
    transposed using block pointers and saving the result to the output tensor.

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
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(N, M),
        strides=(input_stride_y, input_stride_x),
        offsets=(0, 0),
        block_shape=(N, M),
        order=(0, 1),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(N, M),
        strides=(output_stride_x, output_stride_y),
        offsets=(0, 0),
        block_shape=(N, M),
        order=(1, 0),
    )

    input_block = tl.load(input_block_ptr)

    tl.store(output_block_ptr, input_block)


def transpose_with_block_ptr(inputs: torch.Tensor) -> torch.Tensor:
    """Transpose a tensor of shape (M, N) to (N, M) using block pointers. This
    kernel is naive and does not use any blocking, and will not scale up to
    tensors that exceed SRAM capability.

    Args:
        inputs: Tensor of shape (M, N).

    Returns:
        Tensor of shape (N, M) - a transpose of the input tensor.
    """
    M, N = inputs.shape
    outputs = torch.empty((N, M), dtype=inputs.dtype, device=inputs.device)

    transpose_with_block_ptr_kernel[(1, )](
        input_ptr=inputs, output_ptr=outputs,
        M=M, N=N,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
        output_stride_x=outputs.stride(0), output_stride_y=outputs.stride(1),
    )

    return outputs


###############################################################################
# Transpose kernel (blocked)                                                  #
###############################################################################


@triton.jit
def transpose_blocked_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    input_stride_x, input_stride_y,
    output_stride_x, output_stride_y,
):
    """Transpose a tensor of shape (M, N) to (N, M) by loading a row of the
    input tensor and storing it at the correct location in the output tensor.

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
        shape=(N, M),
        strides=(output_stride_x, output_stride_y),
        offsets=(0, program_id),
        block_shape=(N, 1),
        order=(1, 0),
    )

    input_block = tl.load(input_block_ptr)

    tl.store(output_block_ptr, tl.trans(input_block))


def transpose_blocked_row(inputs: torch.Tensor) -> torch.Tensor:
    """Transpose a tensor of shape (M, N) to (N, M) using a blocked copy
    kernel.

    Args:
        inputs: Tensor of shape (M, N).

    Returns:
        Tensor of shape (N, M) - a transpose of the input tensor.
    """
    M, N = inputs.shape
    outputs = torch.empty((N, M), dtype=inputs.dtype, device=inputs.device)

    transpose_blocked_kernel[(M, )](
        input_ptr=inputs, output_ptr=outputs,
        M=M, N=N,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
        output_stride_x=outputs.stride(0), output_stride_y=outputs.stride(1),
    )

    return outputs
