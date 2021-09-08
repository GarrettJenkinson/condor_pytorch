from condor_pytorch.metrics import earth_movers_distance
from condor_pytorch.metrics import ordinal_accuracy
from condor_pytorch.metrics import mean_absolute_error
import pytest
import torch

def test_EMD():
    levels = torch.tensor(
       [[1., 1., 0., 0.],
        [1., 0., 0., 0.],
       [1., 1., 1., 1.]])
    logits = torch.tensor(
       [[2.1, 1.8, -2.1, -1.8],
        [1.9, -1., -1.5, -1.3],
        [1.9, 1.8, 1.7, 1.6]])

    assert torch.allclose(earth_movers_distance(logits,levels),
                          torch.tensor(0.6943),atol=1e-4,rtol=1e-4)

def test_ACC():
   levels = torch.tensor(
       [[1., 1., 0., 0.],
        [1., 0., 0., 0.],
        [1., 1., 1., 1.]])
   logits = torch.tensor(
      [[2.1, 1.8, -2.1, -1.8],
       [1.9, -1., -1.5, -1.3],
       [1.9, 1.8, 1.7, 1.6]])
   assert torch.allclose(ordinal_accuracy(logits, levels),
                         torch.tensor(1.),atol=1e-4,rtol=1e-4)

def test_MAE():
   levels = torch.tensor(
      [[1., 1., 0., 0.],
       [1., 0., 0., 0.],
      [1., 1., 1., 1.]])
   logits = torch.tensor(
      [[2.1, 1.8, -2.1, -1.8],
       [1.9, -1., -1.5, -1.3],
       [1.9, 1.8, 1.7, 1.6]])
   assert torch.allclose(mean_absolute_error(logits, levels),
                         torch.tensor(0.),atol=1e-4,rtol=1e-4)
