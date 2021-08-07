## ordinal_softmax

*ordinal_softmax(x)*

Convert the ordinal logit output to label probabilities.

**Parameters**

x: tf.Tensor, shape=(num_samples,num_classes-1)
    Logit output of the final Dense(num_classes-1) layer.

**Returns**

probs_tensor: tf.Tensor, shape=(num_samples, num_classes)
    Probabilities of each class (columns) for each
    sample (rows).

**Examples**

```
    >>> condor.ordinal_softmax(tf.constant([[-1.,1],[-2,2]]))
    <tf.Tensor: shape=(2, 3), dtype=float32, numpy=
    array([[0.7310586 , 0.07232949, 0.19661194],
    [0.8807971 , 0.01420934, 0.10499357]], dtype=float32)>
```

