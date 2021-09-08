condor_pytorch version: 0.1.0-dev
## CondorOrdinalCrossEntropy

*CondorOrdinalCrossEntropy(logits, levels, importance_weights=None, reduction='mean')*

Computes the CONDOR loss described in

    CONDOR TBD.

**Parameters**

- `logits` : torch.tensor, shape(num_examples, num_classes-1)

    Outputs of the CONDOR layer.


- `levels` : torch.tensor, shape(num_examples, num_classes-1)

    True labels represented as extended binary vectors
    (via `condor_pytorch.dataset.levels_from_labelbatch`).


- `importance_weights` : torch.tensor, shape=(num_classes-1,) (default=None)

    Optional weights for the different labels in levels.
    A tensor of ones, i.e.,
    `torch.ones(num_classes-1, dtype=torch.float32)`
    will result in uniform weights that have the same effect as None.


- `reduction` : str or None (default='mean')

    If 'mean' or 'sum', returns the averaged or summed loss value across
    all data points (rows) in logits. If None, returns a vector of
    shape (num_examples,)

**Returns**

- `loss` : torch.tensor

    A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
    or a loss value for each data record (if `reduction=None`).

**Examples**

```
    >>> import torch
    >>> levels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> CondorOrdinalCrossEntropy(logits, levels)
    tensor(0.8259)
```

