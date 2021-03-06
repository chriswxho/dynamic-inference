import os
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only, sync_ddp

class TensorCheckpoint(Callback):
    
    def __init__(self, every_n_epochs=1, filename='st'):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.filename = filename
        
    def on_validation_start(self, trainer, pl_module):
        if (trainer.current_epoch > 0 and
            (trainer.current_epoch + 1) % self.every_n_epochs == 0 and
            trainer.is_global_zero):
            
            # save tensors here
            path = os.path.join(trainer.log_dir, f'{self.filename}-{trainer.current_epoch}.pt')
            torch.save({'s': pl_module.s, 't': pl_module.t}, path)