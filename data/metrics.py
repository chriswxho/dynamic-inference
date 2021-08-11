import numpy as np
import torch

def SILog(yhat, y, L = 1):
    '''
    yhat: prediction
    y: ground truth
    L: λ in the paper, [0,1]. L=0 gives elementwise L2 loss, 
       L=1 gives scale-invariant loss.
    https://arxiv.org/pdf/1406.2283.pdf
    '''
    idx = ~torch.isnan(y)
    di = torch.log(yhat[idx] + 1e-10) - torch.log(y[idx])
    
    return (di**2).mean() - L * di.mean() ** 2


# In[6]:


def get_metrics(yhat, y, metrics=['absrel', 'delta', 'mae']):
    
    values = []
    idx = ~torch.isnan(y)
    pred = yhat[idx]
    gt = y[idx]
    
    if 'absrel' in metrics:
        values.append((torch.abs(gt - pred) / gt).mean().item())
    if 'delta' in metrics:
        # assume that no values in the ground truth map are zero,
        #  and that infinite/unmapped distances are NaNs
        idx_nonzero = torch.where(torch.where(yhat != 0, 1, 0) * idx)
        delta = torch.max(yhat[idx_nonzero] / y[idx_nonzero], 
                           y[idx_nonzero] / yhat[idx_nonzero])
        values.append(torch.where(delta < 1.25, 1., 0.).mean().item() * 100)
    if 'mae' in metrics:
        values.append((torch.abs(gt - pred)).mean().item())
        
    return np.array(values)