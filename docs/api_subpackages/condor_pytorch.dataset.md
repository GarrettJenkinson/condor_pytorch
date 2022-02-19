condor_pytorch version: 1.0.0
## logits_to_label

*logits_to_label(logits)*

Converts predicted logits from extended binary format
    to integer class labels

**Parameters**

- `logits` : torch.tensor, shape(n_examples, n_labels-1)

    Torch tensor consisting of probabilities returned by ORCA model.

**Examples**

```
    >>> # 3 training examples, 6 classes
    >>> logits = torch.tensor([[ 0.934, -0.861,  0.323, -0.492, -0.295],
    ...                        [-0.496,  0.485,  0.267,  0.124, -0.058],
    ...                        [ 0.985,  0.967, -0.920,  0.819, -0.506]])
    >>> logits_to_label(logits)
    tensor([1, 0, 2])
```

## label_to_levels

*label_to_levels(label, num_classes, dtype=torch.float32)*

Converts integer class label to extended binary label vector

**Parameters**

- `label` : int

    Class label to be converted into a extended
    binary vector. Should be smaller than num_classes-1.


- `num_classes` : int

    The number of class clabels in the dataset. Assumes
    class labels start at 0. Determines the size of the
    output vector.


- `dtype` : torch data type (default=torch.float32)

    Data type of the torch output vector for the
    extended binary labels.

**Returns**

- `levels` : torch.tensor, shape=(num_classes-1,)

    Extended binary label vector. Type is determined
    by the `dtype` parameter.

**Examples**

```
    >>> label_to_levels(0, num_classes=5)
    tensor([0., 0., 0., 0.])
    >>> label_to_levels(1, num_classes=5)
    tensor([1., 0., 0., 0.])
    >>> label_to_levels(3, num_classes=5)
    tensor([1., 1., 1., 0.])
    >>> label_to_levels(4, num_classes=5)
    tensor([1., 1., 1., 1.])
```

## proba_to_label

*proba_to_label(probas)*

Converts predicted probabilities from extended binary format
    to integer class labels

**Parameters**

- `probas` : torch.tensor, shape(n_examples, n_labels)

    Torch tensor consisting of probabilities returned by CORAL model.

**Examples**

```
    >>> # 3 training examples, 6 classes
    >>> probas = torch.tensor([[0.934, 0.861, 0.323, 0.492, 0.295],
    ...                        [0.496, 0.485, 0.267, 0.124, 0.058],
    ...                        [0.985, 0.967, 0.920, 0.819, 0.506]])
    >>> proba_to_label(probas)
    tensor([2, 0, 5])
```

## levels_from_labelbatch

*levels_from_labelbatch(labels, num_classes, dtype=torch.float32)*

Converts a list of integer class label to extended binary label vectors

**Parameters**

- `labels` : list or 1D orch.tensor, shape=(num_labels,)

    A list or 1D torch.tensor with integer class labels
    to be converted into extended binary label vectors.


- `num_classes` : int

    The number of class clabels in the dataset. Assumes
    class labels start at 0. Determines the size of the
    output vector.


- `dtype` : torch data type (default=torch.float32)

    Data type of the torch output vector for the
    extended binary labels.

**Returns**

- `levels` : torch.tensor, shape=(num_labels, num_classes-1)


**Examples**

```
    >>> levels_from_labelbatch(labels=[2, 1, 4], num_classes=5)
    tensor([[1., 1., 0., 0.],
    [1., 0., 0., 0.],
    [1., 1., 1., 1.]])
```

