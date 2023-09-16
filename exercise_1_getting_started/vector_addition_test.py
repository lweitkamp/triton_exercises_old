import pytest

import torch

from exercise_1_getting_started.vector_addition import vector_addition


@pytest.mark.parametrize("N", [16, 32, 64, 128])
def test_vector_addition(N: int):
    a = torch.randn(N, device='cuda')
    b = torch.randn(N, device='cuda')
    outputs = vector_addition(a, b)
    torch.testing.assert_close(outputs, a + b)
