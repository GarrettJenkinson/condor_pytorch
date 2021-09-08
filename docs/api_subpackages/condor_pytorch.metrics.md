condor_pytorch version: 0.1.0-dev
## ordinal_accuracy

*ordinal_accuracy(logits, levels, tolerance=0, reduction='mean')*

Computes the accuracy with a tolerance for ordinal error.

**Parameters**

- `logits` : torch.tensor, shape(num_examples, num_classes-1)

    Outputs of the CONDOR layer.


- `levels` : torch.tensor, shape(num_examples, num_classes-1)

    True labels represented as extended binary vectors
    (via `condor_pytorch.dataset.levels_from_labelbatch`).


- `tolerance` : integer

    Allowed error in the ordinal ranks that will count as
    a correct prediction.


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
    >>> ordinal_accuracy(logits, levels)
    tensor(1.)
```

## mean_absolute_error

*mean_absolute_error(logits, levels, reduction='mean')*

Computes the mean absolute error of ordinal predictions.

**Parameters**

- `logits` : torch.tensor, shape(num_examples, num_classes-1)

    Outputs of the CONDOR layer.


- `levels` : torch.tensor, shape(num_examples, num_classes-1)

    True labels represented as extended binary vectors
    (via `condor_pytorch.dataset.levels_from_labelbatch`).


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
    >>> mean_absolute_error(logits, levels)
    tensor(0.)
```

## earth_movers_distance

*earth_movers_distance(logits, levels, reduction='mean')*

Computes the Earth Movers Distance

**Parameters**

- `logits` : torch.tensor, shape(num_examples, num_classes-1)

    Outputs of the CONDOR layer.


- `levels` : torch.tensor, shape(num_examples, num_classes-1)

    True labels represented as extended binary vectors
    (via `condor_pytorch.dataset.levels_from_labelbatch`).


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
    >>> earth_movers_distance(logits, levels)
    tensor(0.6943)
```

