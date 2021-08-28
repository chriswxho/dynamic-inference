import numpy as np
import torch
import torch.nn.functional as F

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


class DepthMetrics:
    def __init__(self, n_deltas=1, threshold=1.25):
        '''
        n_deltas: number of delta acc metrics to compute
        '''
        self.n_deltas = n_deltas
        self.threshold = threshold
        self.depth_cap = 1e8
    
    def compute_scale_and_shift(self, disp):
        mask = ~torch.isnan(disp)
        shift = torch.median(disp[mask])
        scale = (disp[mask] - shift).abs().mean()
        return scale, shift
    
    def __call__(self, prediction, target, st: tuple=None):
        '''
        Returns absrel, mae, and delta accuracies.
        assume that no values in the ground truth map are zero,
        and that infinite/unmapped distances are NaNs.
        
        assumes that prediction & target are both depth maps.
        '''
    
        # transform predicted disparity to aligned depth
        metrics = {}
        mask = (~torch.isnan(target) * (target != 0)).float()

        if st is None:
            sg, tg = self.compute_scale_and_shift(target)
            sp, tp = self.compute_scale_and_shift(prediction)
            scale = F.relu(sg/sp)
            shift = (tg - tp * scale)
            mode = 'train_'
        else:
            scale, shift = st
            mode = 'val_'
            
        prediction_aligned = scale * prediction + shift

        # optional depthcap step
        prediction_aligned[prediction_aligned > self.depth_cap] = self.depth_cap

        # absrel
        metrics[f'{mode}absrel'] = (torch.abs(prediction_aligned[mask == 1] - target[mask == 1]) / target[mask == 1]).mean().item()

        # mae
        metrics[f'{mode}mae'] = torch.abs(prediction_aligned[mask == 1] - target[mask == 1]).mean().item()
        
        # delta acc
        for delta in range(self.n_deltas):
            acc = torch.zeros_like(prediction_aligned, dtype=torch.float)

            acc[mask == 1] = torch.max(
                prediction_aligned[mask == 1] / target[mask == 1],
                target[mask == 1] / prediction_aligned[mask == 1],
            ).float()

            acc[mask == 1] = (acc[mask == 1] < (self.threshold if delta == 0 
                                                else np.power(self.threshold, delta+1))).float()

            p = torch.sum(acc, (1, 2)) / torch.sum(mask, (1, 2))
        
            metrics[f'{mode}delta{delta+1}'] = torch.mean(p).item() # max acc is 1
            
        if st is None:
            metrics['s'] = scale
            metrics['t'] = shift
            
        return metrics     