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
    def on_save_checkpoint(self, trainer, pl_module, ckpt_dict):
        print('process on rank:', trainer.global_rank)
        if (self.every_n_epochs > 0 
            and ((trainer.current_epoch + 1) % self.every_n_epochs) == 0 
            and trainer.global_rank == 0):
            
            print('attempt save on rank:', trainer.global_rank)
            # save tensors here
            path = os.path.join(trainer.log_dir, f'{self.filename}-{trainer.current_epoch}.pt')
            s,t = sync_ddp(pl_module.s, reduce_op='avg'), sync_ddp(pl_module.t, reduce_op='avg')
            ckpt_dict['s'], ckpt_dict['t'] = s,t
            
            return ckpt_dict