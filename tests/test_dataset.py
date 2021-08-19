from condor_pytorch.dataset import label_to_levels
from condor_pytorch.dataset import levels_from_labelbatch
from condor_pytorch.dataset import proba_to_label
from condor_pytorch.dataset import logits_to_label
import pytest
import torch


def test_label_to_levels():
    assert torch.allclose(label_to_levels(0, num_classes=5),
                          torch.tensor([0., 0., 0., 0.]))
    assert torch.allclose(label_to_levels(1, num_classes=5),
                          torch.tensor([1., 0., 0., 0.]))
    assert torch.allclose(label_to_levels(2, num_classes=5),
                          torch.tensor([1., 1., 0., 0.]))
    assert torch.allclose(label_to_levels(3, num_classes=5),
                          torch.tensor([1., 1., 1., 0.]))
    assert torch.allclose(label_to_levels(4, num_classes=5),
                          torch.tensor([1., 1., 1., 1.]))

def test_levels_from_labelbatch():
    assert torch.allclose(levels_from_labelbatch(labels=[2, 1, 4],
                                                 num_classes=5),
                          torch.tensor([[1., 1., 0., 0.],
                                        [1., 0., 0., 0.],
                                        [1., 1., 1., 1.]]))

def test_proba_to_label():
    probas = torch.tensor([[0.934, 0.861, 0.323, 0.492, 0.295],
                           [0.496, 0.485, 0.267, 0.124, 0.058],
                           [0.985, 0.967, 0.920, 0.819, 0.506]])
    assert torch.allclose(proba_to_label(probas),
                         torch.tensor([2, 0, 5]))

def test_logits_to_label():
    logits = torch.tensor([[ 0.934, -0.861,  0.323, -0.492, -0.295],
                           [-0.496,  0.485,  0.267,  0.124, -0.058],
                           [ 0.985,  0.967, -0.920,  0.819, -0.506]])
    assert torch.allclose(logits_to_label(logits),
                         torch.tensor([1, 0, 2]))

