#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import glob
import time
from datetime import timedelta
import cv2
import numpy as np
import pandas as pd
import argparse

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from dpt.models import DPTDepthModel
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from data.InteriorNetDataset import InteriorNetDataset
from data.metrics import SILog, get_metrics
from util.gpu_config import get_batch_size

torch.manual_seed(0)
np.random.seed(0)

class InteriorNetDPT(pl.LightningModule):
    def __init__(self, batch_size, lr, num_epochs, model_path, s, t):
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

    
def train(lr: float, num_epochs: int, test_mode: bool, checkpoint_path: str, k8s: bool):
    '''
    Run the main training script for DPT.
    -------------------------------------
    lr : learning rate
    num_epochs : number of epochs to train for
    test_mode : if True, trains on a small subset of the data
    checkpoint_path : path to checkpoint to restore training progress
    k8s : if True, acts as if running on k8s
    '''

#     checkpoint_path = '/christh9-pvc/train-logs/finetune/version_5/checkpoints/dpt-finetune-epoch=9.ckpt'
    # k8s paths
    k8s_repo = r'opt/repo/dynamic-inference'
    k8s_pvc = r'christh9-pvc'

    # path settings
    input_path = 'input'
    output_path = 'output_monodepth'
    model_path = 'weights/dpt_hybrid_nyu-2ce69ec7.pt'
    dataset_path = 'video_inference_common/resources'
    logs_path = 'train-logs'

    if k8s:
        input_path = os.path.join(k8s_repo, input_path)
        output_path = os.path.join(k8s_repo, output_path)
        model_path = os.path.join(k8s_pvc, 'dpt-hybrid-nyu.pt')
        dataset_path = os.path.join(k8s_repo, dataset_path)
        logs_dir = os.path.join(k8s_pvc, logs_path)
        os.chdir('/')

    net_w = 640
    net_h = 480

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

    transform = Compose(
        [
            Resize(
                net_w,
                net_h,
                resize_target=None,
                keep_aspect_ratio=True,
                ensure_multiple_of=32,
                resize_method="minimal",
                image_interpolation_method=cv2.INTER_CUBIC,
            ),
            normalization,
            PrepareForNet(),
        ]
    )
    
    # create dataloader

    start = time.time()

    batch_size = get_batch_size()

    print('-- Hyperparams --')
    print(f'Batchsize: {batch_size}')
    print(f'Learning rate: {lr}')
    print(f'Epochs: {num_epochs}')
    print('-----------------')

    # get shifted statistics

    get_new = False

    if get_new:
        start = time.time()

        full_dataset = InteriorNetDataset(dataset_path, transform=transform)
        d = []
        p = 0.1
        for i in np.random.choice(len(full_dataset), size=round(p*len(full_dataset)), replace=False):
            d.append(full_dataset[i]['depth'].flatten())

        d = 1 / np.array(d)
        idx = ~np.isnan(d)
        t = np.median(d[idx])
        s = (d[idx] - t).mean()

        print(f'Retrieved statistics in {timedelta(seconds=round(time.time()-start,2))}s')
        print(f's: {s}, t: {t}')

    else:
        s, t = 0.4364, 0.4115

    # depth stats:
    # s: 0.22012925148010254, t: 2.445845127105713, 
    # s: 0.20440419018268585, t: 2.446396827697754

    # disparity stats:
    # s: 0.4363926351070404, t: 0.4114949703216553

    # original nyu stats:
    # s: 0.000305, t: 0.1378

    start = time.time()

    # model setup
    model = InteriorNetDPT(batch_size, lr, num_epochs, model_path, s, t)
    # logging setup
    logger = TensorBoardLogger(logs_dir, 
                               name='finetune',
                               log_graph=True)

    # dataloader setup
    interiornet_dataset = InteriorNetDataset(dataset_path, transform=transform, subsample=test_mode)
    dataloader = DataLoader(interiornet_dataset, 
                            batch_size=model.hparams.batch_size, 
                            shuffle=True,
                            prefetch_factor=8,
                            num_workers=4*torch.cuda.device_count() if torch.cuda.is_available() else 0)

    # checkpointing
    checkpoint = ModelCheckpoint(every_n_epochs=num_epochs//10,
                                 save_on_train_epoch_end=True,
                                 save_top_k=-1,
                                 filename='dpt-finetune-{epoch}')


    print(f'Created datasets in {timedelta(seconds=round(time.time()-start,2))}')
    
    if checkpoint_path is not None:
        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path,
                                     gpus=torch.cuda.device_count(), 
                                     max_epochs=model.hparams.num_epochs,
                                     accelerator='ddp',
                                     logger=logger,
                                     callbacks=[checkpoint])
            else:
                trainer = pl.Trainer(resume_from_checkpoint=checkpoint_path,
                                     gpus=torch.cuda.device_count(), 
                                     max_epochs=model.hparams.num_epochs,
                                     logger=logger,
                                     callbacks=[checkpoint])
        else:
            trainer = pl.Trainer(max_epochs=1, logger=logger)
    else:

        if torch.cuda.is_available():
            if torch.cuda.device_count() > 1:
                trainer = pl.Trainer(gpus=torch.cuda.device_count(), 
                                     max_epochs=model.hparams.num_epochs,
                                     accelerator='ddp',
                                     logger=logger,
                                     callbacks=[checkpoint])
            else:
                trainer = pl.Trainer(gpus=torch.cuda.device_count(), 
                                     max_epochs=model.hparams.num_epochs,
                                     logger=logger,
                                     callbacks=[checkpoint])
        else:
            trainer = pl.Trainer(max_epochs=1, logger=logger)
    
    print('Training')

    try:    
        start = time.time()
        trainer.fit(model, dataloader)

    except Exception as e:
        print('Training was halted due to the following error:')
        print(e)

    else:
        print(f'Training completed in {timedelta(seconds=round(time.time()-start,2))}')

    finally:
        print(f'Training checkpoints and logs are saved in {trainer.log_dir}')
        exp_idx = len(list(filter(lambda f: '.pt' in f, os.listdir(os.path.join(logs_dir)))))
        print(f'Final trained weights saved in finetune{exp_idx}.pt')

    # eval this video if test_mode=True:
    # 3FO4IW2QC9U7_original_1_1

    torch.save(model.state_dict(), os.path.join(logs_dir, f'finetune{exp_idx}.pt'))

    logger.save()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train the DPT on the InteriorNet dataset.')

    parser.add_argument(
        '-l', '--lr', default=1e-5, help='learning rate', type=float
    )

    parser.add_argument(
        '-e', '--epochs', default=100, help='num. of epochs', type=int
    )
    
    parser.add_argument(
        '-t', '--test', action='store_true'
    )

    parser.add_argument(
        '-c', '--checkpoint', default=None, help='path to a checkpoint to resume'
    )
    
    parser.add_argument(
        '-k', '--k8s', action='store_true'
    )
    args = parser.parse_args()
    
    train(args.lr, args.epochs, args.test, args.checkpoint, args.k8s)