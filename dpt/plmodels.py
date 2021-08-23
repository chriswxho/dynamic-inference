import torch
import torch.optim as optim

import pytorch_lightning as pl
from dpt.models import DPTDepthModel
from data.metrics import SILog, DepthMetrics

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
#         self.example_input_array = torch.ones((1, 3, net_h, net_w))
            
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
        
        
        self.log_dict(self.metrics(yhat.detach(), y.detach()),
                      on_step=False,
                      on_epoch=True,
                      sync_dist=torch.cuda.device_count() > 1)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        yhat = self.model(x)
        loss = SILog(yhat, y)
        self.log('val_loss', loss, 
                 on_step=False,
                 on_epoch=True, 
                 sync_dist=torch.cuda.device_count() > 1)
        
        self.log_dict(self.metrics(yhat, y),
                      on_step=False,
                      on_epoch=True,
                      sync_dist=torch.cuda.device_count() > 1)
        
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        
    def on_train_epoch_start(self):
        if 'verbose' in kwargs and not kwargs['verbose']:
            print(f'Epoch {self.current_epoch}')
        
#     def training_epoch_end(self, _):
#         self.logger.log_graph(self)
    
    def configure_optimizers(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), 
                          lr=self.hparams.lr)