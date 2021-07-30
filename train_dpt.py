#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import cv2
import numpy as np

import util.io

import torch
import torch.optim as optim
from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

import time
import datetime
# In[2]:

torch.manual_seed(0)

# k8s paths
k8s = True
k8s_repo = r'opt/repo/dynamic-inference'
k8s_pvc = r'../../christh9-pvc'

# path settings
input_path = 'input'
output_path = 'output_monodepth'
model_path = 'weights/dpt_hybrid_nyu-2ce69ec7.pt'

if k8s:
    input_path = os.path.join(k8s_repo, input_path)
    output_path = os.path.join(k8s_repo, output_path)
    model_path = os.path.join(k8s_pvc, 'dpt-hybrid-nyu.pt')
#     script_output = os.path.join(k8s_pvc, 'dpt-timings', f'runtimes-{device_name}.csv')


# In[3]:


# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

model_type = 'dpt_hybrid_nyu'

# load network
if model_type == "dpt_large":  # DPT-Large
    net_w = net_h = 384
    model = DPTDepthModel(
        path=model_path,
        backbone="vitl16_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
elif model_type == "dpt_hybrid":  # DPT-Hybrid
    net_w = net_h = 384
    model = DPTDepthModel(
        path=model_path,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )
    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
elif model_type == "dpt_hybrid_kitti":
    net_w = 1216
    net_h = 352

    model = DPTDepthModel(
        path=model_path,
        scale=0.00006016,
        shift=0.00579,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
elif model_type == "dpt_hybrid_nyu":
    net_w = 640
    net_h = 480

    model = DPTDepthModel(
        path=model_path,
        scale=0.000305,
        shift=0.1378,
        invert=True,
        backbone="vitb_rn50_384",
        non_negative=True,
        enable_attention_hooks=False,
    )

    normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
elif model_type == "midas_v21":  # Convolutional model
    net_w = net_h = 384

    model = MidasNet_large(model_path, non_negative=True)
    normalization = NormalizeImage(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
else:
    assert (
        False
    ), f"model_type '{model_type}' not implemented, use: --model_type [dpt_large|dpt_hybrid|dpt_hybrid_kitti|dpt_hybrid_nyu|midas_v21]"
    
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

model.to(device)

print('Loaded model')

# In[4]:


# figure out how to implement this in training, don't use this for now

# if optimize == True and device == torch.device("cuda"):
#     model = model.to(memory_format=torch.channels_last)
#     model = model.half()


# In[10]:


import re
import itertools
from torch.utils.data import Dataset, DataLoader

from video_inference_common.video_inference.datasets import interiornet

def getlines(files: [str], subsample):
    ''' helper function to get stripped lines from multiple files'''
    names = []
    # todo: delete after verification
    if not subsample:
        for f in files:
            names.append(map(lambda x: x.strip(), open(f).readlines()))
        return list(itertools.chain.from_iterable(names))

    else:
        for f in files:
            names.append(open(f).readline().strip())
            break
            
        return names

class InteriorNetDataset(Dataset):
    def __init__(self, dataset_path: str, train=True, transform=None, subsample=False):
        '''
        dataset_path: path to the folder containing the txts that specify dataset
                      (relative to ./dynamic-inference)
        train: specify to use the training or test split
        transform: optional transform to be applied per sample
        subsample: take a subsample of all the training data
        '''
        subsets = re.compile(f'.*?({"train" if train else "test"}).*?')
        video_names = map(lambda p: os.path.join(dataset_path, p), 
                          filter(subsets.match, os.listdir(dataset_path)))
        self.videos = np.array(getlines(video_names, subsample))
        self.transform = transform
        self.path = dataset_path
        
    def __len__(self):
        return 1000 * len(self.videos) # each video is 1000 frames
    
    def __getitem__(self, idx):
        
        # idx will come as video_index * frame_index
        img_name = self.videos[idx // 1000]
        frame_idx = idx % 1000
        
        im = interiornet.read_rgb(img_name, frame_idx)
        depth = interiornet.read_depth(img_name, frame_idx)
        
        if self.transform:
            im = self.transform({'image': im})['image']
        
        return {'image': im, 'depth': depth}


# In[5]:


def SILogLoss(yhat, y, L = 1):
    '''
    yhat: prediction
    y: ground truth
    L: λ in the paper, [0,1]. L=0 gives elementwise L2 loss, 
       L=1 gives scale-invariant loss.
    https://arxiv.org/pdf/1406.2283.pdf
    '''
    di = torch.log(yhat[idx] + 1) - torch.log(y[idx] + 1)
    
    return (di**2).mean() - L * di.mean() ** 2


# In[6]:


def get_metrics(yhat, y, metrics=['absrel', 'delta', 'mae']):
    
    yhat.detach()
    y.detach()
    
    values = []
    idx = ~torch.isnan(y)
    pred = yhat[idx]
    gt = y[idx]
    
    if 'absrel' in metrics:
        values.append((torch.abs(gt - pred) / gt).mean())
    if 'delta' in metrics:
        thresh = torch.max(pred/gt, gt/pred)
        values.append(thresh[thresh < 1.25].mean())
    if 'mae' in metrics:
        values.append((torch.abs(gt - pred)).mean())
        
    return np.array(values)


# In[7]:


model.load_state_dict(torch.load(model_path))
print('Loaded model weights')

# In[8]:

start = time.time()


dataset_path = 'video_inference_common/resources'
if k8s:
    dataset_path = os.path.join(k8s_repo, dataset_path)
    
batch_size = 1
print(f'Batchsize: {batch_size}')

interiornet_dataset = InteriorNetDataset(dataset_path, transform=transform, subsample=True)
dataloader = DataLoader(interiornet_dataset, 
                        batch_size=batch_size, 
                        shuffle=True, 
                        num_workers=0)


print(f'Created datasets in {round(time.time()-start,3)}s')
# In[13]:

start = time.time()
# get shifted statistics
full_dataset = InteriorNetDataset(dataset_path, transform=transform)
d = []
p = 0.1
for i in np.random.choice(len(full_dataset), size=round(p*len(full_dataset)), replace=False):
    d.append(full_dataset[i]['depth'].flatten())

d = np.array(d)
idx = ~np.isnan(d)
t = np.median(d[idx])
s = (d[idx] - t).mean()


print(f'Retrieved statistics in {round(time.time()-start,3)}s')
print(f's: {s}, t: {t}')


# In[ ]:


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
    print(f'Step {step+1}')
    for sample in dataloader:
        x, y = sample['image'].to(device), sample['depth'].to(device)
        yhat = model(x)
        loss = SILogLoss((yhat - t) / s, 1/y)
        loss.backward()
        running_loss += loss.detach().item() * x.size(0)
        num_samples += x.size(0)
        
        metrics.append(get_metrics(yhat, y))
        
        optimizer.step()
        optimizer.zero_grad()
    
    losses.append(running_loss / num_samples if num_samples > 0 else running_loss)
    print(f'SILogLoss: {losses[-1]}')
    print(f'AbsRel: {metrics[-1][0]} \t MAE: {metrics[-1][2]}')
    print(f'Step time: {str(datetime.timedelta(round(seconds=time.time() - start)))}')


# data saving

import pandas as pd

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

