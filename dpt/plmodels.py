from collections import Counter
from itertools import chain

import os
import math
import numpy as np
import torch
import torch.optim as optim

import pytorch_lightning as pl
from dpt.models import DPTDepthModel
from data.metrics import SILog, DepthMetrics
from util.misc import round_sig

class InteriorNetDPT(pl.LightningModule):
    
    def __init__(self, batch_size: int, lr: float, num_epochs: int, model_path: str, 
                 s=1, t=0, net_w=640, net_h=480, **kwargs):
        
        super().__init__()
        self.model = DPTDepthModel(
                        path=model_path,
                        scale=s,
                        shift=t,
                        invert=True,
                        backbone="vitb_rn50_384",
                        non_negative=True,
                        enable_attention_hooks=False,
                     )
        
        self.num_epochs = num_epochs
        self.model.pretrained.model.patch_embed.requires_grad = False
        
        self.save_hyperparameters()
        self.metrics = DepthMetrics()
        self.kwargs = kwargs
        
        self.s = 1581.6136
        self.t = 0.9938
        
        self.val_outputs = None
#         self.example_input_array = torch.ones((1, 3, net_h, net_w))
            
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        yhat = self.model(x)
        
        metrics = self.metrics(yhat.detach(), y.detach(), (self.s, self.t))
        
#         self.s.append(metrics.pop('s'))
#         self.t.append(metrics.pop('t'))
        
#         self.print({'s': self.s[-1], 't': self.t[-1]})
        
        for metric,val in metrics.items():
            self.log(metric, val,
                     on_step=False, 
                     on_epoch=True,
                     rank_zero_only=True,
                     sync_dist=torch.cuda.device_count() > 1)
        
        loss = SILog(self.s * yhat + self.t, y)
        self.log('train_loss', loss, 
                 on_step=False,
                 on_epoch=True,
                 rank_zero_only=True,
                 sync_dist=torch.cuda.device_count() > 1)
        
        # gather scale, shift from computed metrics
        # to use for validation later
            
#         self.log_dict(metrics,
#                       on_step=False,
#                       on_epoch=True,
#                       sync_dist=torch.cuda.device_count() > 1)
        
        return {'loss': loss, **metrics}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        yhat = self.model(x)
        
        metrics = self.metrics(yhat, y, (self.s, self.t))
        
        for metric,val in metrics.items():
            self.log(metric, val,
                     on_step=False, 
                     on_epoch=True,
                     rank_zero_only=True,
                     sync_dist=torch.cuda.device_count() > 1)
            
#         self.log_dict(metrics,
#                       on_step=False,
#                       on_epoch=True,
#                       sync_dist=torch.cuda.device_count() > 1)
        
        loss = SILog(self.s * yhat + self.t, y)
        self.log('val_loss', loss, 
                 on_step=False,
                 on_epoch=True, 
                 rank_zero_only=True,
                 sync_dist=torch.cuda.device_count() > 1)
            
        return {'loss': loss, **metrics}
    
    def configure_optimizers(self):
        return optim.Adam([
                            {'params': filter(lambda p: p.requires_grad, self.model.pretrained.parameters())},
                            {'params': self.model.scratch.parameters(), 'lr': self.hparams.lr * 10}
                          ], 
                          lr=self.hparams.lr)
    
    def training_epoch_end(self, epoch_outputs):
        if 'interactive' not in self.kwargs or not self.kwargs['interactive']:
            res = Counter()
            for out in epoch_outputs:
                res += out
            self.print(f'--- Epoch {self.current_epoch} training ---')
            for name, val in res.items():
                if name == 'loss':
                    name = 'train_loss'
                    val = val.item()
                if math.isinf(val):
                    self.print(f'{name}: {val}')
                    continue
                self.print(f'{name}: {round_sig(val / len(epoch_outputs), 4)}')
            self.print('-'*25)
            self.logger.log_graph(self)
        
    def validation_epoch_end(self, epoch_outputs):
        self.val_outputs = epoch_outputs
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        
    def on_train_epoch_start(self):
#         self.s, self.t = [], []
        if 'interactive' not in self.kwargs or not self.kwargs['interactive']:
            self.print(f'Epoch {self.current_epoch}')
            
    def on_train_epoch_end(self):
        if 'interactive' not in self.kwargs or not self.kwargs['interactive']:
            res = Counter()
            for out in self.val_outputs:
                res += out
            self.print(f'--- Epoch {self.current_epoch} validation ---')
            for name, val in res.items():
                if name == 'loss': 
                    name = 'val_loss'
                    val = val.item()
                if math.isinf(val):
                    self.print(f'{name}: {val}')
                    continue
                self.print(f'{name}: {round_sig(val / len(self.val_outputs), 4)}')
            self.print('-'*25)
            self.logger.log_graph(self)

        self.val_outputs = None
            
#     def on_validation_start(self):
#         if len(self.s) > 0:
#             self.s, self.t = torch.tensor(self.s).type_as(self.s[-1]).mean(0), torch.tensor(self.t).type_as(self.t[-1]).mean(0)
#         else:
#             raise ValueError('Empty s,t arrays (empty batches)')
            
    def on_after_backward(self):
        pvcpath = '/christh9-pvc/train-logs/finetune/grads'
        if 'grads' not in os.listdir(os.path.dirname(pvcpath)):
            os.mkdir(pvcpath)
            
        grads = {}
        for i,param in enumerate(self.model.parameters()):
            grads[i] = param.grad
        torch.save(grads, os.path.join(pvcpath,f'grads-single.pt'))
#         torch.save(grads, os.path.join(pvcpath,f'grads-single-{self.global_rank}.pt'))