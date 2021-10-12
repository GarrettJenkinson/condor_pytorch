import torch
import torch.nn.functional as F

def ordinal_softmax(x, device='cpu'):
    """ Convert the ordinal logit output to label probabilities.

    Parameters
    ----------
    x: torch.Tensor, shape=(num_samples,num_classes-1)
        Logit output of the final Dense(num_classes-1) layer.
        
    device: 'cpu', 'cuda', or None (default='cpu')
        If GPUs are utilized, then the device should be passed accordingly. 

    Returns
    ----------
    probs_tensor: torch.Tensor, shape=(num_samples, num_classes)
        Probabilities of each class (columns) for each
        sample (rows).

    Examples
    ----------
    >>> ordinal_softmax(torch.tensor([[-1.,1],[-2,2]]))
    tensor([[0.7311, 0.0723, 0.1966],
            [0.8808, 0.0142, 0.1050]])
    """
    
    # Convert the ordinal logits into cumulative probabilities.
    
    
    log_probs = F.logsigmoid(x).to(device)
    cum_probs = torch.cat((torch.ones(x.shape[0],1,dtype=torch.float32).to(device),
                         torch.exp(torch.cumsum(log_probs, dim = 1)),
                         torch.zeros(x.shape[0],1,dtype=torch.float32).to(device)),
                        dim=1)    
    return cum_probs[:,0:-1] - cum_probs[:,1:]
