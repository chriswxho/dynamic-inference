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

from dpt.plmodels import InteriorNetDPT
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from data.InteriorNetDataset import InteriorNetDataset
from util.gpu_config import get_batch_size

torch.manual_seed(0)
np.random.seed(0)

    
def train(lr: float, batch_size: int, num_epochs: int, other_args):
    '''
    Run the main training script for DPT.
    -------------------------------------
    lr : learning rate
    batch_size: batch size (across all GPUs s.t. the total batchsize = batch_size)
    num_epochs : number of epochs to train for
    
    other_args:
    -------
    test : if True, trains on a small subset of the data
    checkpoint_path : path to checkpoint to restore training progress
    k8s : if True, acts as if running on k8s
    verbose: if True, gives live loading bar updates, otherwise prints last epoch #
    '''

    # k8s paths
    k8s_repo = r'opt/repo/dynamic-inference'
    k8s_pvc = r'christh9-pvc'

    # path settings
    input_path = 'input'
    output_path = 'output_monodepth'
    model_path = 'weights/dpt_hybrid_nyu-2ce69ec7.pt'
    dataset_path = 'video_inference_common/resources'
    logs_path = 'train-logs'

    if other_args['k8s']:
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

    start = time.time()

    batch_size = get_batch_size(batch_size)

    print('-- Hyperparams --')
    print(f'Batchsize: {batch_size}')
    print(f'Learning rate: {lr}')
    print(f'Epochs: {num_epochs}')
    print('-----------------')

    start = time.time()

    # model setup
    model = InteriorNetDPT(batch_size, lr, num_epochs, model_path, verbose=other_args['verbose'])
    
    # logging setup
    logger = TensorBoardLogger(logs_dir, 
                               name='finetune',
                               log_graph=True)

    # dataloader setup
    train_dataset = InteriorNetDataset(dataset_path, split='train' if not other_args['test'] else 'test', 
                                       transform=transform, subsample=other_args['test'])
    
    val_dataset = InteriorNetDataset(dataset_path, split='val' if not other_args['test'] else 'test',
                                     transform=transform, subsample=other_args['test'])
    
    train_loader = DataLoader(train_dataset, 
                              batch_size=model.hparams.batch_size, 
                              shuffle=True,
                              prefetch_factor=16, # increase or decrease based on free gpu mem
                              num_workers=4*torch.cuda.device_count() if torch.cuda.is_available() else 0)
    
    val_loader = DataLoader(val_dataset,
                            batch_size=model.hparams.batch_size,
                            prefetch_factor=16, # increase or decrease based on free gpu mem
                            num_workers=4*torch.cuda.device_count() if torch.cuda.is_available() else 0)

    # checkpointing
    checkpoint = ModelCheckpoint(every_n_epochs=5,
                                 save_on_train_epoch_end=True,
                                 save_top_k=-1,
                                 filename='dpt-finetune-{epoch}')


    print(f'Created datasets in {timedelta(seconds=round(time.time()-start,2))}')
    
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            trainer = pl.Trainer(resume_from_checkpoint=path if (path := other_args['checkpoint']) else None,
                                 gpus=torch.cuda.device_count(), 
                                 max_epochs=model.hparams.num_epochs,
                                 accelerator='ddp',
                                 logger=logger,
                                 callbacks=[checkpoint] if not other_args['test'] else None,
                                 num_sanity_val_steps=0,
                                 progress_bar_refresh_rate=None if other_args['verbose'] else 0)
        else:
            trainer = pl.Trainer(resume_from_checkpoint=path if (path := other_args['checkpoint']) else None,
                                 gpus=1, 
                                 max_epochs=model.hparams.num_epochs,
                                 logger=logger,
                                 callbacks=[checkpoint] if not other_args['test'] else None,
                                 num_sanity_val_steps=0,
                                 progress_bar_refresh_rate=None if other_args['verbose'] else 0)
    else:
        trainer = pl.Trainer(max_epochs=1, logger=logger)
    
    print('Training')

    try:    
        start = time.time()
        trainer.fit(model, 
                    train_dataloaders=train_loader,
                    val_dataloaders=val_loader)

    except Exception as e:
        print('Training was halted due to the following error:')
        raise

    else:
        print(f'Training completed in {timedelta(seconds=round(time.time()-start,2))}')

    finally:
        print(f'Training checkpoints and logs are saved in {trainer.log_dir}')
        exp_idx = len(list(filter(lambda f: '.pt' in f, os.listdir(os.path.join(logs_dir)))))
        print(f'Final trained weights saved in finetune{exp_idx}.pt')
        torch.save(model.state_dict(), os.path.join(logs_dir, f'finetune{exp_idx}.pt'))

    logger.save()
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Train the DPT on the InteriorNet dataset.')

    parser.add_argument(
        '-l', '--lr', default=1e-5, help='learning rate', type=float
    )
    
    parser.add_argument(
        '-b', '--batchsize', default=None, help='batchsize', type=int
    )

    parser.add_argument(
        '-e', '--epochs', default=100, help='num. of epochs', type=int
    )
    
    parser.add_argument(
        '-t', '--test', action='store_true'
    )
    # trains on the first 20 frames of this video if test=True:
    # 3FO4IW2QC9U7_original_1_1

    parser.add_argument(
        '-c', '--checkpoint', default=None, help='path to a checkpoint to resume'
    )
    
    parser.add_argument(
        '-k', '--k8s', action='store_true'
    )
    
    parser.add_argument(
        '-v', '--verbose', action='store_true'
    )
    args = parser.parse_args()
    
    other_args = dict(vars(args))
    del other_args['lr'], other_args['batchsize'], other_args['epochs']

    train(args.lr, args.batchsize, args.epochs, other_args)