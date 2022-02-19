condor_pytorch version: 1.0.0
## condor_negloglikeloss

*condor_negloglikeloss(logits, labels, reduction='mean')*

computes the negative log likelihood loss described in

    condor tbd.

**parameters**

- `logits` : torch.tensor, shape(num_examples, num_classes-1)

    outputs of the condor layer.


- `labels` : torch.tensor, shape(num_examples, num_classes-1)

    true labels represented as extended binary vectors
    (via `condor_pytorch.dataset.levels_from_labelbatch`).


- `reduction` : str or none (default='mean')

    if 'mean' or 'sum', returns the averaged or summed loss value across
    all data points (rows) in logits. if none, returns a vector of
    shape (num_examples,)

**returns**

- `loss` : torch.tensor

    a torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
    or a loss value for each data record (if `reduction=none`).

**examples**

```
    >>> import torch
    >>> labels = torch.tensor(
    ...    [[1., 1., 0., 0.],
    ...     [1., 0., 0., 0.],
    ...    [1., 1., 1., 1.]])
    >>> logits = torch.tensor(
    ...    [[2.1, 1.8, -2.1, -1.8],
    ...     [1.9, -1., -1.5, -1.3],
    ...     [1.9, 1.8, 1.7, 1.6]])
    >>> condor_negloglikeloss(logits, labels)
    tensor(0.4936)
```

## CondorOrdinalCrossEntropy

*CondorOrdinalCrossEntropy(logits, levels, importance_weights=None, reduction='mean')*

computes the condor loss described in

    condor tbd.

**parameters**

- `logits` : torch.tensor, shape(num_examples, num_classes-1)

    outputs of the condor layer.


- `levels` : torch.tensor, shape(num_examples, num_classes-1)

    true labels represented as extended binary vectors
    (via `condor_pytorch.dataset.levels_from_labelbatch`).


- `importance_weights` : torch.tensor, shape=(num_classes-1,) (default=none)

    optional weights for the different labels in levels.
    a tensor of ones, i.e.,
    `torch.ones(num_classes-1, dtype=torch.float32)`
    will result in uniform weights that have the same effect as none.


- `reduction` : str or none (default='mean')

    if 'mean' or 'sum', returns the averaged or summed loss value across
    all data points (rows) in logits. if none, returns a vector of
    shape (num_examples,)

**returns**

- `loss` : torch.tensor

    a torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
    or a loss value for each data record (if `reduction=none`).

**examples**

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

