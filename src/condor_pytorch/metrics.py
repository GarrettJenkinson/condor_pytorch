import torch.nn.functional as F
import torch
from .activations import ordinal_softmax

def earth_movers_distance(logits, levels, device='cpu', reduction='mean'):
    """Computes the Earth Movers Distance

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CONDOR layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `condor_pytorch.dataset.levels_from_labelbatch`).
        
    device: 'cpu', 'cuda', or None (default='cpu')
        If GPUs are utilized, then the device should be passed accordingly. 

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    Examples
    ----------
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
    """
    nclasses = logits.shape[1]+1
    nbatch = logits.shape[0]
    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                         % (logits.shape, levels.shape))

    cum_probs = ordinal_softmax(logits, device)
    y_true = torch.sum(levels,dim=1,keepdim=True,dtype=logits.dtype)

#     y_dist = torch.abs(torch.tile(y_true,(1,nclasses))
#                        -torch.tile(torch.arange(0,nclasses),(nbatch,1)))
    y_dist = torch.abs(y_true.repeat(1,nclasses) - torch.arange(0,nclasses).repeat(nbatch,1)).to(device)
    
    val = torch.sum(torch.mul(cum_probs,y_dist),1)

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss

def ordinal_accuracy(logits, levels, device='cpu', tolerance=0, reduction='mean'):
    """Computes the accuracy with a tolerance for ordinal error.

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CONDOR layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `condor_pytorch.dataset.levels_from_labelbatch`).
    
    device: 'cpu', 'cuda', or None (default='cpu')
        If GPUs are utilized, then the device should be passed accordingly. 
    
    tolerance : integer
        Allowed error in the ordinal ranks that will count as
        a correct prediction.

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    Examples
    ----------
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
    """
    nclasses = logits.shape[1]+1
    nbatch = logits.shape[0]
    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                         % (logits.shape, levels.shape))

    y_true = torch.sum(levels,dim=1,keepdim=True,dtype=logits.dtype).to(device)

    y_est = torch.sum(torch.cumprod(torch.sigmoid(logits),dim=1)>0.5,dim=1,keepdim=True,dtype=logits.dtype).to(device)
    
    # 1 when correct and 0 else
    val = torch.le(torch.abs(y_true-y_est),tolerance).to(torch.float32)

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss

def mean_absolute_error(logits, levels, reduction='mean'):
    """Computes the mean absolute error of ordinal predictions.

    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CONDOR layer.

    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `condor_pytorch.dataset.levels_from_labelbatch`).

    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)

    Returns
    ----------
        loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).

    Examples
    ----------
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
    """
    nclasses = logits.shape[1]+1
    nbatch = logits.shape[0]
    if not logits.shape == levels.shape:
        raise ValueError("Please ensure that logits (%s) has the same shape as levels (%s). "
                         % (logits.shape, levels.shape))

    y_true = torch.sum(levels,dim=1,keepdim=True,dtype=logits.dtype)

    y_est = torch.sum(torch.cumprod(torch.sigmoid(logits),dim=1)>0.5,dim=1,keepdim=True,dtype=logits.dtype)

    # 1 when correct and 0 else
    val = torch.abs(y_true-y_est)

    if reduction == 'mean':
        loss = torch.mean(val)
    elif reduction == 'sum':
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = ('Invalid value for `reduction`. Should be "mean", '
             '"sum", or None. Got %s' % reduction)
        raise ValueError(s)

    return loss
