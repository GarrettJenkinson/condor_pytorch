condor_pytorch version: 1.0.0
## ordinal_accuracy

*ordinal_accuracy(logits, levels, device='cpu', tolerance=0, reduction='mean')*

Computes the accuracy with a tolerance for ordinal error.

**Parameters**

- `logits` : torch.tensor, shape(num_examples, num_classes-1)

    Outputs of the CONDOR layer.


- `levels` : torch.tensor, shape(num_examples, num_classes-1)

    True labels represented as extended binary vectors
    (via `condor_pytorch.dataset.levels_from_labelbatch`).

    device: 'cpu', 'cuda', or None (default='cpu')
    If GPUs are utilized, then the device should be passed accordingly.


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

*earth_movers_distance(logits, levels, device='cpu', reduction='mean')*

Computes the Earth Movers Distance

**Parameters**

- `logits` : torch.tensor, shape(num_examples, num_classes-1)

    Outputs of the CONDOR layer.


- `levels` : torch.tensor, shape(num_examples, num_classes-1)

    True labels represented as extended binary vectors
    (via `condor_pytorch.dataset.levels_from_labelbatch`).

    device: 'cpu', 'cuda', or None (default='cpu')
    If GPUs are utilized, then the device should be passed accordingly.


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

