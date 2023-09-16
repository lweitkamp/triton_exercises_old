import pytest

import torch

from exercise_1_getting_started.transpose import (
    transpose, transpose_with_block_ptr, transpose_blocked_row)


@pytest.mark.parametrize("M, N", [(16, 16), (32, 16)])
def test_transpose(M: int, N: int):
    inputs = torch.randn((M, N), device='cuda')
    outputs = transpose(inputs)
    torch.testing.assert_close(inputs.T, outputs)


@pytest.mark.parametrize("M, N", [(16, 16), (32, 16)])
def test_transpose_with_block_ptr(M: int, N: int):
    inputs = torch.randn((M, N), device='cuda')
    outputs = transpose_with_block_ptr(inputs)
    torch.testing.assert_close(inputs.T, outputs)


@pytest.mark.parametrize("M, N", [(16, 16), (32, 16)])
def test_transpose_blocked_row(M: int, N: int):
    inputs = torch.randn((M, N), device='cuda')
    outputs = transpose_blocked_row(inputs)
    torch.testing.assert_close(inputs.T, outputs)
