from collections import Counter
from itertools import chain

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
        
        self.val_outputs = None
#         self.example_input_array = torch.ones((1, 3, net_h, net_w))

        # s,t values
    
        self.st = {
                    10:  (1.2606, -0.45781),
                    50:  (1.2566, -0.45565), 
                    100: (1.2534, -0.45245)
                  }
            
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        yhat = self.model(x)
        loss = SILog(yhat, y)
        self.log('train_loss', loss, 
                 on_step=False,
                 on_epoch=True, 
                 sync_dist=torch.cuda.device_count() > 1)
        
        s,t = self.st[100]
        yhat = s * yhat.detach() + t
        
        metrics = self.metrics(yhat, y.detach(), mode='train')
        
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=torch.cuda.device_count() > 1)
        
        return {'loss': loss, **metrics}
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        yhat = self.model(x)
        loss = SILog(yhat, y)
        self.log('val_loss', loss, 
                 on_step=False,
                 on_epoch=True, 
                 sync_dist=torch.cuda.device_count() > 1)
        
        s,t = self.st[100]
        yhat = s * yhat + t
        
        metrics = self.metrics(yhat, y, mode='val')
        self.log_dict(metrics,
                      on_step=False,
                      on_epoch=True,
                      sync_dist=torch.cuda.device_count() > 1)
            
        return {'loss': loss, **metrics}
    
    def configure_optimizers(self):
        return optim.Adam([
                            {'params': filter(lambda p: p.requires_grad, self.model.pretrained.parameters())},
                            {'params': self.model.scratch.parameters(), 'lr': self.hparams.lr * 10}
                          ], 
                          lr=self.hparams.lr)
    
    def training_epoch_end(self, epoch_outputs):
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
        if 'verbose' in self.kwargs and not self.kwargs['verbose']:
            self.print(f'Epoch {self.current_epoch}')
            
    def on_train_epoch_end(self):
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