import torch
import torch.optim as optim

import pytorch_lightning as pl
from dpt.models import DPTDepthModel
from data.metrics import SILog, get_metrics

class InteriorNetDPT(pl.LightningModule):
    def __init__(self, batch_size: int, lr: float, num_epochs: int, model_path: str, s=1, t=0):
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
        
#         self.example_input_array = torch.ones((3, net_h, net_w)) # try this next
            
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        yhat = self.model(x)
        loss = SILog(yhat, y)
        self.log('train_loss', loss, 
                 on_epoch=True, 
                 sync_dist=torch.cuda.device_count() > 1)
        
        metrics = get_metrics(yhat.detach(), y.detach())
        self.log('absrel', metrics[0],
                 on_step=False,
                 on_epoch=True, 
                 sync_dist=torch.cuda.device_count() > 1)
        self.log('delta_acc', metrics[1], 
                 on_step=False,
                 on_epoch=True, 
                 sync_dist=torch.cuda.device_count() > 1)
        self.log('mae', metrics[2], 
                 on_step=False,
                 on_epoch=True, 
                 sync_dist=torch.cuda.device_count() > 1)
        return loss
    
    def on_train_start(self):
        self.logger.log_hyperparams(self.hparams)
        
#     def training_epoch_end(self, _):
#         self.logger.log_graph(self)
    
    def configure_optimizers(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), 
                          lr=self.hparams.lr)