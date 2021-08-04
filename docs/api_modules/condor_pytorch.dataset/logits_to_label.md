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

