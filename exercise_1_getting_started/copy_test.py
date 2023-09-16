import pytest

import torch

from exercise_1_getting_started.copy import copy_1d, copy, copy_blocked


@pytest.mark.parametrize("N", [16, 32, 64])
def test_copy_1d(N: int):
    inputs = torch.randn(N, device='cuda')
    outputs = copy_1d(inputs)
    torch.testing.assert_close(inputs, outputs)


@pytest.mark.parametrize("M, N", [(16, 16), (32, 32), (32, 64)])
def test_copy(M: int, N: int):
    inputs = torch.randn(M, N, device='cuda')
    outputs = copy(inputs)
    torch.testing.assert_close(inputs, outputs)


@pytest.mark.parametrize("M, N", [(32, 64), (128, 128)])
def test_copy_blocked(M: int, N: int):
    inputs = torch.randn(M, N, device='cuda')
    outputs = copy_blocked(inputs)
    torch.testing.assert_close(inputs, outputs)
