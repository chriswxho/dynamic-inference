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

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from dataset.InteriorNetDataset import InteriorNetDataset

import util.io

# In[2]:

torch.manual_seed(0)
np.random.seed(0)

# k8s paths
k8s = True
k8s_repo = r'opt/repo/dynamic-inference'
k8s_pvc = r'../../christh9-pvc'

# path settings
input_path = 'input'
output_path = 'output_monodepth'
model_path = 'weights/dpt_hybrid_nyu-2ce69ec7.pt'
dataset_path = 'video_inference_common/resources'

if k8s:
    input_path = os.path.join(k8s_repo, input_path)
    output_path = os.path.join(k8s_repo, output_path)
    model_path = os.path.join(k8s_pvc, 'dpt-hybrid-nyu.pt')
    dataset_path = os.path.join(k8s_repo, dataset_path)
#     script_output = os.path.join(k8s_pvc, 'dpt-timings', f'runtimes-{device_name}.csv')

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
    
batch_size = 8
print(f'Batchsize: {batch_size}')

interiornet_dataset = InteriorNetDataset(dataset_path, transform=transform, subsample=True)
dataloader = DataLoader(interiornet_dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=0)


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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

model = DPTDepthModel(
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

model.to(device)

print('Loaded model')

# figure out how to implement this in training, don't use this for now

# if optimize == True and device == torch.device("cuda"):
#     model = model.to(memory_format=torch.channels_last)
#     model = model.half()

def SILogLoss(yhat, y, L = 1):
    '''
    yhat: prediction
    y: ground truth
    L: λ in the paper, [0,1]. L=0 gives elementwise L2 loss, 
       L=1 gives scale-invariant loss.
    https://arxiv.org/pdf/1406.2283.pdf
    '''
    idx = ~torch.isnan(y)
    di = torch.log(yhat[idx] + 1) - torch.log(y[idx] + 1)
    
    return (di**2).mean() - L * di.mean() ** 2


# In[6]:


def get_metrics(yhat, y, metrics=['absrel', 'delta', 'mae']):
    
    values = []
    idx = ~torch.isnan(y)
    pred = yhat[idx]
    gt = y[idx]
    
    if 'absrel' in metrics:
        values.append((torch.abs(gt - pred) / gt).mean().item())
    if 'delta' in metrics:
        # assume that no values in the ground truth map are zero,
        #  and that infinite/unmapped distances are NaNs
        idx_nonzero = torch.where(torch.where(yhat != 0, 1, 0) * idx)
        delta = torch.max(yhat[idx_nonzero] / y[idx_nonzero], 
                           y[idx_nonzero] / yhat[idx_nonzero])
        values.append(torch.where(delta < 1.25, 1., 0.).mean().item() * 100)
    if 'mae' in metrics:
        values.append((torch.abs(gt - pred)).mean().item())
        
    return np.array(values)


model.load_state_dict(torch.load(model_path))
print('Loaded model weights')


print('Training')

num_steps = 10
losses = []
metrics = []
lr = 1e-5

optimizer = optim.Adam(model.parameters(), lr)


for step in range(num_steps):
    
    start = time.time()
    running_loss = 0.0
    num_samples = 0
    running_metrics = []
    
    print(f'Step {step+1}')
    for sample in dataloader:
        x, y = sample['image'].to(device), sample['depth'].to(device)
        yhat = model(x)
        
        if torch.any(torch.isnan(yhat)):
            print('ERROR: NaNs in model output')
            
        loss = SILogLoss(yhat, 1/y)
        loss.backward()
        
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        
        running_metrics.append(get_metrics(yhat.detach(), 1/y.detach()))
        
        num_samples += x.detach().size(0)
        running_loss += loss.detach().item() * x.detach().size(0)
    
    losses.append(running_loss / num_samples if num_samples > 0 else 0)
    
    running_metrics = np.array(running_metrics)

    metrics.append(running_metrics.mean(axis=0))
    
    print(f'SILogLoss: {losses[-1]}')
    print(f'AbsRel: {metrics[-1][0]}\tDelta acc%: {metrics[-1][1]}\tMAE: {metrics[-1][2]}')
    print(f'Step time: {str(datetime.timedelta(seconds=round(time.time() - start)))}')


# data saving

losses = np.array(losses)
metrics = np.array(metrics)

logs_dir = os.path.join(k8s_pvc, 'train-logs')

if not os.path.exists(logs_dir):
    os.mkdir(logs_dir)
    
df = pd.DataFrame({
                   'loss': losses, 
                   'absrel': metrics[:,0],
                   'delta': metrics[:,1],
                   'mae': metrics[:,2]
                  })

df.to_csv(os.path.join(logs_dir, 'testrun.csv'))

torch.save(model.state_dict(), os.path.join(logs_dir, 'finetune.pt'))

