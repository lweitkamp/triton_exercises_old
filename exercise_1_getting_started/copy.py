import torch

import triton
import triton.language as tl


###############################################################################
# Copy (1D) Kernel (naive)                                                    #
###############################################################################


@triton.jit
def copy_1d_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    N: tl.constexpr,
):
    """Copy the values of a vector to the same location in another vector. This
    kernel is naive and does not use any blocking, and will not scale up to
    tensors that exceed SRAM capability.

    Args:
        input_ptr: Pointer to the input vector.
        output_ptr: Pointer to the output vector.
        N: Number of elements in the input vector.
    """
    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(...),
        strides=(...),
        offsets=(...),
        block_shape=(...),
        order=(0, ),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(...),
        strides=(...),
        offsets=(...),
        block_shape=(...),
        order=(0, ),
    )

    input_block = tl.load(...)

    tl.store(...)


def copy_1d(inputs: torch.Tensor) -> torch.Tensor:
    """Copy a vector using a naive copy kernel. This kernel is naive and does
    not use any blocking, and will not scale up to tensors that exceed SRAM
    capability.

    Args:
        inputs: Tensor of shape (M, N).

    Returns:
        Tensor of shape (M, N) - a copy of the input tensor.
    """
    N = inputs.shape[0]
    outputs = torch.empty((N, ), dtype=inputs.dtype, device=inputs.device)

    copy_1d_kernel[(1, )](input_ptr=inputs, output_ptr=outputs, N=N)

    return outputs


###############################################################################
# Copy (2D) Kernel (naive)                                                    #
###############################################################################


@triton.jit
def copy_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    input_stride_x, input_stride_y,
    output_stride_x, output_stride_y,
):
    """Copy the values of a tensor to the same location in another tensor.

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
        shape=...,
        strides=(...),
        offsets=(...),
        block_shape=(...),
        order=(1, 0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=...,
        strides=(...),
        offsets=(...),
        block_shape=(...),
        order=(1, 0),
    )

    input_block = tl.load(...)

    tl.store(...)


def copy(inputs: torch.Tensor) -> torch.Tensor:
    """Copy a tensor using a naive copy kernel. This kernel is naive and does
    not use any blocking, and will not scale up to tensors that exceed SRAM
    capability.

    Args:
        inputs: Tensor of shape (M, N).

    Returns:
        Tensor of shape (M, N) - a copy of the input tensor.
    """
    M, N = inputs.shape
    outputs = torch.empty((M, N), dtype=inputs.dtype, device=inputs.device)

    copy_kernel[(1, )](
        input_ptr=inputs, output_ptr=outputs,
        M=M, N=N,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
        output_stride_x=outputs.stride(0), output_stride_y=outputs.stride(1),
    )

    return outputs


###############################################################################
# Copy (2D) Kernel (blocked)                                                  #
###############################################################################


@triton.jit
def copy_blocked_kernel(
    input_ptr: tl.tensor, output_ptr: tl.tensor,
    M: tl.constexpr, N: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    input_stride_x, input_stride_y,
    output_stride_x, output_stride_y,
):
    """Copy a block of a tensor to the same location in another tensor.

    Args:
        input_ptr: Pointer to the input tensor.
        output_ptr: Pointer to the output tensor.
        M: Number of rows in the input tensor.
        N: Number of columns in the input tensor.
        BLOCK_M: Block size along the row dim.
        BLOCK_N: Block size along the column dim.
        input_stride_x: Stride of the input tensor along the row dim.
        input_stride_y: Stride of the input tensor along the column dim.
        output_stride_x: Stride of the output tensor along the row dim.
        output_stride_y: Stride of the output tensor along the column dim.
    """
    offset_M = ...
    offset_N = ...

    input_block_ptr = tl.make_block_ptr(
        base=input_ptr,
        shape=(...),
        strides=(...),
        offsets=(...),
        block_shape=(...),
        order=(1, 0),
    )
    output_block_ptr = tl.make_block_ptr(
        base=output_ptr,
        shape=(...),
        strides=(...),
        offsets=(...),
        block_shape=(...),
        order=(1, 0),
    )

    input_block = tl.load(...)

    tl.store(...)


def copy_blocked(inputs: torch.Tensor) -> torch.Tensor:
    """Copy a tensor using a blocked copy kernel.

    Args:
        inputs: Tensor of shape (M, N).

    Returns:
        Tensor of shape (M, N) - a copy of the input tensor.
    """
    M, N = inputs.shape
    outputs = torch.empty((M, N), dtype=inputs.dtype, device=inputs.device)

    BLOCK_M, BLOCK_N = 16, 16

    copy_blocked_kernel[(triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))](
        input_ptr=inputs, output_ptr=outputs,
        M=M, N=N,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        input_stride_x=inputs.stride(0), input_stride_y=inputs.stride(1),
        output_stride_x=outputs.stride(0), output_stride_y=outputs.stride(1),
    )

    return outputs
