import os
import torch
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.utilities.distributed import rank_zero_only, sync_ddp

class TensorCheckpoint(Callback):
    
    def __init__(self, every_n_epochs=1, filename='st'):
        super().__init__()
        self.every_n_epochs = every_n_epochs
        self.filename = filename
        
    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, ckpt):
        print('callback executed on rank', trainer.global_rank)
        if (self.every_n_epochs > 0 
            and ((trainer.current_epoch + 1) % self.every_n_epochs) == 0 
            and trainer.global_rank == 0):
            
            # save tensors here
            path = os.path.join(trainer.log_dir, f'{self.filename}-{trainer.current_epoch}.pt')
            torch.save({'s': pl_module.s, 't': pl_module.t}, path)
    
        return ckpt