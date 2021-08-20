from condor_pytorch.losses import CondorOrdinalCrossEntropy
import pytest
import torch

def test_CondorOrdinalCrossEntropy():
    levels = torch.tensor(
       [[1., 1., 0., 0.],
        [1., 0., 0., 0.],
       [1., 1., 1., 1.]])
    logits = torch.tensor(
       [[2.1, 1.8, -2.1, -1.8],
        [1.9, -1., -1.5, -1.3],
        [1.9, 1.8, 1.7, 1.6]])

    assert torch.allclose(earth_movers_distance(logits,levels),
                          torch.tensor(0.6943))
