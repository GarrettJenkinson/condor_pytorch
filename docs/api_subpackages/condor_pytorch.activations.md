condor_pytorch version: 1.0.0
## ordinal_softmax

*ordinal_softmax(x, device='cpu')*

Convert the ordinal logit output to label probabilities.

**Parameters**

x: torch.Tensor, shape=(num_samples,num_classes-1)
    Logit output of the final Dense(num_classes-1) layer.

    device: 'cpu', 'cuda', or None (default='cpu')
    If GPUs are utilized, then the device should be passed accordingly.

**Returns**

probs_tensor: torch.Tensor, shape=(num_samples, num_classes)
    Probabilities of each class (columns) for each
    sample (rows).

**Examples**

```
    >>> ordinal_softmax(torch.tensor([[-1.,1],[-2,2]]))
    tensor([[0.7311, 0.0723, 0.1966],
    [0.8808, 0.0142, 0.1050]])
```

