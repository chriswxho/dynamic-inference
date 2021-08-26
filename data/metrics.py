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


class DepthMetrics:
    def __init__(self, n_deltas=1, threshold=1.25):
        '''
        n_deltas: number of delta acc metrics to compute
        '''
        self.n_deltas = n_deltas
        self.threshold = threshold
        self.depth_cap = 1e8
    
    def compute_scale_and_shift(self, prediction, target, mask):
        
        # dealing with NaN in target:
        target[mask == 0] = 0
        
        # system matrix: A = [[a_00, a_01], [a_10, a_11]]
        a_00 = torch.sum(mask * prediction * prediction, (1, 2))
        a_01 = torch.sum(mask * prediction, (1, 2))
        a_11 = torch.sum(mask, (1, 2))

        # right hand side: b = [b_0, b_1]
        b_0 = torch.sum(mask * prediction * target, (1, 2))
        b_1 = torch.sum(mask * target, (1, 2))

        # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
        s = torch.zeros_like(b_0)
        t = torch.zeros_like(b_1)
        
        det = a_00 * a_11 - a_01 * a_01
        # A needs to be a positive definite matrix.
        valid = det > 0

        s[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
        t[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]
        
#         if torch.any(torch.isnan(torch.cat([s,t]))):
#             print('a_00 finite?', torch.all(torch.isfinite(a_00)))
#             print('a_01 finite?', torch.all(torch.isfinite(a_01)))
#             print('a_11 finite?', torch.all(torch.isfinite(a_11)))
#             print()
#             print('b_0 finite?', torch.all(torch.isfinite(b_0)))
#             print('b_1 finite?', torch.all(torch.isfinite(b_1)))
#             print()
#             print('det finite and nonzero?', torch.all(torch.isfinite(det[valid]) * torch.is_nonzero(det[valid])))
#             print('is s[valid] good?', torch.all(torch.isfinite(s[valid])))
#             print('is t[valid] good?', torch.all(torch.isfinite(t[valid])))

        return s, t
    
    def __call__(self, prediction, target, mode: str):
        '''
        Returns absrel, mae, and delta accuracies.
        assume that no values in the ground truth map are zero,
        and that infinite/unmapped distances are NaNs.
        
        assumes that prediction & target are both depth maps.
        '''
    
        # transform predicted disparity to aligned depth
        metrics = {}
        mask = ~torch.isnan(target) * (target != 0)

        mode += '_'

        # optional depthcap step
        prediction[prediction > self.depth_cap] = self.depth_cap

        # absrel
        metrics[f'{mode}absrel'] = (torch.abs(prediction[mask == 1] - target[mask == 1]) / target[mask == 1]).mean().item()

        # mae
        metrics[f'{mode}mae'] = torch.abs(prediction[mask == 1] - target[mask == 1]).mean().item()
        
        # delta acc
        for delta in range(self.n_deltas):
            acc = torch.zeros_like(prediction, dtype=torch.float)

            acc[mask == 1] = torch.max(
                prediction[mask == 1] / target[mask == 1],
                target[mask == 1] / prediction[mask == 1],
            ).float()

            acc[mask == 1] = (acc[mask == 1] < (self.threshold if delta == 0 
                                                else np.power(self.threshold, delta+1))).float()

            p = torch.sum(acc, (1, 2)) / torch.sum(mask, (1, 2))
        
            metrics[f'{mode}delta{delta+1}'] = torch.mean(p).item() # max acc is 1
            
        return metrics     