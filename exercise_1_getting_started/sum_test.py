import pytest

import torch

from exercise_1_getting_started.sum import sum_blocked_row


@pytest.mark.parametrize("M, N", [(16, 16), (32, 16)])
def test_sum_row(M: int, N: int):
    inputs = torch.randn((M, N), device='cuda')
    outputs = sum_blocked_row(inputs)
    torch.testing.assert_close(inputs.sum(dim=1), outputs)
