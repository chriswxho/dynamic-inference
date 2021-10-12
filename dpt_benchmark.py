#!/usr/bin/env python
# coding: utf-8

import os
import glob
import time
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torchvision.transforms import Compose

from dpt.transforms import Resize, NormalizeImage, PrepareForNet

from dpt.plmodels import InteriorNetDPT
from data.InteriorNetDataset import InteriorNetDataset
from data.metrics import SILog, DepthMetrics

# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu'
print(device_name)
    
# weights setting
weights = 'nyu'

# k8s paths
k8s = True
k8s_repo = r'opt/repo/dynamic-inference'
k8s_pvc = r'christh9-pvc'

# path settings
input_path = 'input'
output_path = 'output_monodepth'
model_path = 'weights/dpt_hybrid_nyu-2ce69ec7.pt' if weights == 'nyu' else 'weights/dpt_hybrid-midas-501f0c75.pt'

if k8s:
    input_path = os.path.join(k8s_repo, input_path)
    output_path = os.path.join(k8s_repo, output_path)
    model_path = os.path.join(k8s_pvc, 'dpt-hybrid-nyu.pt' if weights == 'nyu' else 'dpt-hybrid-midas.pt')
    script_output = os.path.join(k8s_pvc, 'dpt-timings', f'runtimes-embed-{device_name}.csv')
    os.chdir('/')

runs = 500
timings = np.zeros((runs))

# load network

# model = InteriorNetDPT(batch_size=1, 
#                        lr=0, 
#                        num_epochs=0, 
#                        model_path=checkpoint_path)

# model.to(device)
# model.freeze()


# # get input
# img_names = glob.glob(os.path.join(input_path, "*"))
# num_images = len(img_names)

# # create output folder
# os.makedirs(output_path, exist_ok=True)


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

import timm
model = timm.create_model("vit_base_resnet50_384", pretrained=True)
model.to(device)
model.eval()

# input
img = np.random.rand(480,640,3)

img_input = transform({"image": img})["image"] 

sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

# start, end = None, None
# if device == torch.device('cuda'):
#     start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

# custom decoder step for CUDA timing
# def decoder(model, x):
    
#     start, end = None, None
#     if torch.cuda.is_available(): # assumes availability = in use
#         start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
#     b, c, h, w = x.shape

#     if torch.cuda.is_available():
#         start.record()
#     else:
#         start = time.time()
        
#     layer_1 = model.pretrained.activations["1"]
#     layer_2 = model.pretrained.activations["2"]
#     layer_3 = model.pretrained.activations["3"]
#     layer_4 = model.pretrained.activations["4"]

#     layer_1 = model.pretrained.act_postprocess1[0:2](layer_1)
#     layer_2 = model.pretrained.act_postprocess2[0:2](layer_2)
#     layer_3 = model.pretrained.act_postprocess3[0:2](layer_3)
#     layer_4 = model.pretrained.act_postprocess4[0:2](layer_4)

#     unflatten = nn.Sequential(
#         nn.Unflatten(
#             2,
#             torch.Size(
#                 [
#                     h // model.pretrained.model.patch_size[1],
#                     w // model.pretrained.model.patch_size[0],
#                 ]
#             ),
#         )
#     )

#     if layer_1.ndim == 3:
#         layer_1 = unflatten(layer_1)
#     if layer_2.ndim == 3:
#         layer_2 = unflatten(layer_2)
#     if layer_3.ndim == 3:
#         layer_3 = unflatten(layer_3)
#     if layer_4.ndim == 3:
#         layer_4 = unflatten(layer_4)

#     layer_1 = model.pretrained.act_postprocess1[3 : len(model.pretrained.act_postprocess1)](layer_1)
#     layer_2 = model.pretrained.act_postprocess2[3 : len(model.pretrained.act_postprocess2)](layer_2)
#     layer_3 = model.pretrained.act_postprocess3[3 : len(model.pretrained.act_postprocess3)](layer_3)
#     layer_4 = model.pretrained.act_postprocess4[3 : len(model.pretrained.act_postprocess4)](layer_4)

#     layer_1_rn = model.scratch.layer1_rn(layer_1)
#     layer_2_rn = model.scratch.layer2_rn(layer_2)
#     layer_3_rn = model.scratch.layer3_rn(layer_3)
#     layer_4_rn = model.scratch.layer4_rn(layer_4)

#     path_4 = model.scratch.refinenet4(layer_4_rn)
#     path_3 = model.scratch.refinenet3(path_4, layer_3_rn)
#     path_2 = model.scratch.refinenet2(path_3, layer_2_rn)
#     path_1 = model.scratch.refinenet1(path_2, layer_1_rn)

#     out = model.scratch.output_conv(path_1)
    
#     elapsed = None
    
#     if torch.cuda.is_available():
#         end.record()
#         torch.cuda.synchronize()
#         elapsed = start.elapsed_time(end)
#     else:
#         end = time.time()
#         elapsed = end - start

#     return out, elapsed


# In[9]:
def embed_forward(x):
    if torch.cuda.is_available():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
        start.record()
        model.patch_embed(x)
        end.record()
    
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
    else:
        start = time.time()
        model.patch_embed(x)
        end = time.time()
        elapsed = end - start
    
    return elapsed

for _ in range(5):
    _ = embed_forward(sample)
    
with torch.no_grad():
    for r in range(runs):
         timings[r] = embed_forward(sample)
        
# # warm up devices
# for _ in range(5):
#     _ = model.forward(sample)

# # compute
# with torch.no_grad():
#     for r in range(runs):
#         # time encoder
#         if device == torch.device('cuda'):
#             start.record()
#             _ = model.pretrained.model.forward_flex(sample)
#             end.record()
            
#             torch.cuda.synchronize()
#             elapsed = start.elapsed_time(end)
#             timings[r,0] = elapsed
            
#         else:
#             start = time.time()
#             _ = model.pretrained.model.forward_flex(sample)
#             end = time.time()
#             timings[r,0] = end - start
            
#         # time decoder
#         _, elapsed = decoder(model, sample)
#         timings[r,1] = elapsed

# torch times in milliseconds, convert to seconds
if torch.cuda.is_available():
    timings = timings / 1000

if k8s:
    os.makedirs(os.path.dirname(script_output), exist_ok=True)
    df = pd.DataFrame({'embed': [timings.mean(), timings.std()],},
#                        'decoder': [timings.mean(), timings.std()]},
                      index=['mean','std'])
    df.to_csv(script_output)
    
print('Mean times:', timings.mean(), timings.mean())
print('Std:', timings.std(), timings.std())

