#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import glob
import time
import datetime
import cv2
import numpy as np
import pandas as pd

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import Compose

import pytorch_lightning as pl
from pytorch_lightning.loggers import CSVLogger

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from data.InteriorNetDataset import InteriorNetDataset
from data.metrics import SILog, get_metrics
from util.gpu_config import get_batch_size

import util.io

# In[2]:

torch.manual_seed(0)
np.random.seed(0)

# k8s paths
k8s = True
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
print(f'Batchsize: {batch_size}')

interiornet_dataset = InteriorNetDataset(dataset_path, transform=transform, subsample=True)
dataloader = DataLoader(interiornet_dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=4*torch.cuda.device_count() if torch.cuda.is_available() else 0)


print(f'Created datasets in {round(time.time()-start,3)}s')

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

    print(f'Retrieved statistics in {round(time.time()-start,3)}s')
    print(f's: {s}, t: {t}')

else:
    s, t = 0.4364, 0.4115

# s: 0.22012925148010254, t: 2.445845127105713, s: 0.20440419018268585, t: 2.446396827697754

# inverse stats:
# s: 0.4363926351070404, t: 0.4114949703216553

# select device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print("device: %s" % device)


# model.to(device)

# print('Loaded model')

# figure out how to implement this in training, don't use this for now

# if optimize == True and device == torch.device("cuda"):
#     model = model.to(memory_format=torch.channels_last)
#     model = model.half()

class InteriorNetDPT(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = DPTDepthModel(
                        path=model_path,
                        scale=s,
                        shift=t,
                        #     scale=0.000305,
                        #     shift=0.1378,
                        invert=True,
                        backbone="vitb_rn50_384",
                        non_negative=True,
                        enable_attention_hooks=False,
                     )
    
    def forward(self):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch['image'], batch['depth']
        yhat = self.model(x)
        loss = SILog(yhat, 1/y)
        self.log('train_loss', loss, on_epoch=True)
        
        metrics = get_metrics(yhat.detach(), 1/y.detach())
        self.log('absrel', metrics[0], on_epoch=True)
        self.log('delta_acc', metrics[1], on_epoch=True)
        self.log('mae', metrics[2], on_epoch=True)
        return loss
    
    def configure_optimizers(self):
        return optim.Adam(filter(lambda p: p.requires_grad, self.parameters()), 
                          lr=1e-5)

num_epochs = 20

model = InteriorNetDPT()

idx = len(os.listdir(logs_dir))
logger = CSVLogger(os.path.join(logs_dir, 'finetune-log'), name=f'dpt-nyu-finetune{idx}')

model.model.pretrained.model.patch_embed.requires_grad = False

if torch.cuda.is_available():
    if torch.cuda.device_count() > 1:
        trainer = pl.Trainer(gpus=torch.cuda.device_count(), 
                             max_epochs=num_epochs,
                             accelerator='ddp',
                             logger=logger,
                             progress_bar_refresh_rate=0)
    else:
        trainer = pl.Trainer(gpus=torch.cuda.device_count(), 
                             max_epochs=num_epochs,
                             logger=logger,
                             progress_bar_refresh_rate=0)
else:
    trainer = pl.Trainer(max_epochs=1, logger=logger)
    
print('Training')

trainer.fit(model, dataloader)

# num_steps = 50
# losses = []
# metrics = []
# lr = 1e-5

# optimizer = optim.Adam(model.parameters(), lr)


# for step in range(num_steps):
    
#     start = time.time()
#     running_loss = 0.0
#     num_samples = 0
#     running_metrics = []
    
#     print(f'Step {step+1}')
#     for sample in dataloader:
#         x, y = sample['image'].to(device), sample['depth'].to(device)
#         yhat = model(x)
            
#         loss = SILog(yhat, 1/y)
#         loss.backward()
        
#         optimizer.step()
#         optimizer.zero_grad(set_to_none=True)
        
#         running_metrics.append(get_metrics(yhat.detach(), 1/y.detach()))
        
#         num_samples += x.detach().size(0)
#         running_loss += loss.detach().item() * x.detach().size(0)
    
#     losses.append(running_loss / num_samples if num_samples > 0 else 0)
    
#     running_metrics = np.array(running_metrics)

#     metrics.append(running_metrics.mean(axis=0))
    
#     print(f'SILogLoss: {losses[-1]}')
#     print(f'AbsRel: {metrics[-1][0]}\tDelta acc%: {metrics[-1][1]}\tMAE: {metrics[-1][2]}')
#     print(f'Step time: {str(datetime.timedelta(seconds=round(time.time() - start)))}')


# # data saving

# losses = np.array(losses)
# metrics = np.array(metrics)

# if not os.path.exists(logs_dir):
#     os.mkdir(logs_dir)
    
# df = pd.DataFrame({
#                    'loss': losses, 
#                    'absrel': metrics[:,0],
#                    'delta': metrics[:,1],
#                    'mae': metrics[:,2]
#                   })

# df.to_csv(os.path.join(logs_dir, 'testrun.csv'))

# eval this video:
# 3FO4IW2QC9U7_original_1_1

torch.save(model.state_dict(), os.path.join(logs_dir, f'finetune{idx}.pt'))

logger.experiment.save()