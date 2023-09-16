import pytest

import torch

from exercise_1_getting_started.reshape import reshape


@pytest.mark.parametrize("M, N, K", [(2, 8, 8)])
def test_reshape(M: int, N: int, K: int):
    inputs = torch.arange(M * N * K, dtype=torch.float32, device='cuda')
    inputs = inputs.reshape((M, N, K))
    outputs = reshape(inputs)
    # Unfortunately there is no real support for view, reshape yet in Triton.
    assert not torch.allclose(inputs.view(M, N * K), outputs)
