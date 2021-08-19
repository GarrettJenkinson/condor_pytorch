from condor_pytorch.activations import ordinal_softmax
import pytest
import torch


def test_ordinal_softmax():
    x = torch.tensor([[-1.,1],[-2,2]])
    res = ordinal_softmax(x)
    expect = torch.tensor([[0.7310586 , 0.07232949, 0.19661194],
                          [0.8807971 , 0.01420934, 0.10499357]])
    assert torch.allclose(res,expect)
