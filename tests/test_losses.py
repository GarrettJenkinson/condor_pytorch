from condor_pytorch.losses import CondorOrdinalCrossEntropy
from condor_pytorch.losses import condor_negloglikeloss
import pytest
import torch


def test_CondorOrdinalCrossEntropy():
    levels = torch.tensor([[1., 1., 0., 0.],
                           [1., 0., 0., 0.],
                           [1., 1., 1., 1.]])
    logits = torch.tensor([[2.1, 1.8, -2.1, -1.8],
                           [1.9, -1., -1.5, -1.3],
                           [1.9, 1.8, 1.7, 1.6]])
    assert torch.allclose(CondorOrdinalCrossEntropy(logits,levels),
                          torch.tensor(0.8259),atol=1e-4,rtol=1e-4)

def test_CondorNLL():
    levels = torch.tensor([[1., 1., 0., 0.],
                           [1., 0., 0., 0.],
                           [1., 1., 1., 1.]])
    logits = torch.tensor([[2.1, 1.8, -2.1, -1.8],
                           [1.9, -1., -1.5, -1.3],
                           [1.9, 1.8, 1.7, 1.6]])
    assert torch.allclose(condor_negloglikeloss(logits,levels),
                          torch.tensor(0.4936),atol=1e-4,rtol=1e-4)
