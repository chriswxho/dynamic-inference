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
import torch.nn as nn
from torchvision.transforms import Compose
import timm

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
    script_output = os.path.join(k8s_pvc, 'dpt-timings', f'runtimes-{"".join(device_name.split())}.csv')
    os.chdir('/')

runs = 500

timings = { 
            'embed': np.zeros(runs),
            'encoder' : np.zeros(runs),
            'fusion' : np.zeros(runs),
            'decoder': np.zeros(runs),
            'total': np.zeros(runs),
          }

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

model = InteriorNetDPT(1, 0, 0, model_path)
model.freeze()
model.cuda()
model = model.model

vit_rn = timm.create_model("vit_base_resnet50_384", pretrained=True)
vit_rn.eval()
vit_rn.cuda()

# input
img = np.random.rand(net_h,net_w,3)

img_input = transform({"image": img})["image"] 

sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

def decoder(model, x):
    
    start, end = None, None
    if torch.cuda.is_available(): # assumes availability = in use
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        
    if torch.cuda.is_available():
        start.record()
    else:
        start = time.time()
    
    b, c, h, w = x.shape
        
    layer_1 = model.pretrained.activations["1"]
    layer_2 = model.pretrained.activations["2"]
    layer_3 = model.pretrained.activations["3"]
    layer_4 = model.pretrained.activations["4"]

    layer_1 = model.pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = model.pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = model.pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = model.pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // model.pretrained.model.patch_size[1],
                    w // model.pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = model.pretrained.act_postprocess1[3 : len(model.pretrained.act_postprocess1)](layer_1)
    layer_2 = model.pretrained.act_postprocess2[3 : len(model.pretrained.act_postprocess2)](layer_2)
    layer_3 = model.pretrained.act_postprocess3[3 : len(model.pretrained.act_postprocess3)](layer_3)
    layer_4 = model.pretrained.act_postprocess4[3 : len(model.pretrained.act_postprocess4)](layer_4)

    layer_1_rn = model.scratch.layer1_rn(layer_1)
    layer_2_rn = model.scratch.layer2_rn(layer_2)
    layer_3_rn = model.scratch.layer3_rn(layer_3)
    layer_4_rn = model.scratch.layer4_rn(layer_4)

    path_4 = model.scratch.refinenet4(layer_4_rn)
    path_3 = model.scratch.refinenet3(path_4, layer_3_rn)
    path_2 = model.scratch.refinenet2(path_3, layer_2_rn)
    path_1 = model.scratch.refinenet1(path_2, layer_1_rn)

    out = model.scratch.output_conv(path_1)
    
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
    else:
        end = time.time()
        elapsed = end - start

    return elapsed

def fusion(model, x):
    
    start, end = None, None
    if torch.cuda.is_available(): # assumes availability = in use
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    b, c, h, w = x.shape
        
    layer_1 = model.pretrained.activations["1"]
    layer_2 = model.pretrained.activations["2"]
    layer_3 = model.pretrained.activations["3"]
    layer_4 = model.pretrained.activations["4"]

    layer_1 = model.pretrained.act_postprocess1[0:2](layer_1)
    layer_2 = model.pretrained.act_postprocess2[0:2](layer_2)
    layer_3 = model.pretrained.act_postprocess3[0:2](layer_3)
    layer_4 = model.pretrained.act_postprocess4[0:2](layer_4)

    unflatten = nn.Sequential(
        nn.Unflatten(
            2,
            torch.Size(
                [
                    h // model.pretrained.model.patch_size[1],
                    w // model.pretrained.model.patch_size[0],
                ]
            ),
        )
    )

    if layer_1.ndim == 3:
        layer_1 = unflatten(layer_1)
    if layer_2.ndim == 3:
        layer_2 = unflatten(layer_2)
    if layer_3.ndim == 3:
        layer_3 = unflatten(layer_3)
    if layer_4.ndim == 3:
        layer_4 = unflatten(layer_4)

    layer_1 = model.pretrained.act_postprocess1[3 : len(model.pretrained.act_postprocess1)](layer_1)
    layer_2 = model.pretrained.act_postprocess2[3 : len(model.pretrained.act_postprocess2)](layer_2)
    layer_3 = model.pretrained.act_postprocess3[3 : len(model.pretrained.act_postprocess3)](layer_3)
    layer_4 = model.pretrained.act_postprocess4[3 : len(model.pretrained.act_postprocess4)](layer_4)

    layer_1_rn = model.scratch.layer1_rn(layer_1)
    layer_2_rn = model.scratch.layer2_rn(layer_2)
    layer_3_rn = model.scratch.layer3_rn(layer_3)
    layer_4_rn = model.scratch.layer4_rn(layer_4)
    
    if torch.cuda.is_available():
        start.record()
    else:
        start = time.time()

    path_4 = model.scratch.refinenet4(layer_4_rn)
    path_3 = model.scratch.refinenet3(path_4, layer_3_rn)
    path_2 = model.scratch.refinenet2(path_3, layer_2_rn)
    path_1 = model.scratch.refinenet1(path_2, layer_1_rn)
    
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
    else:
        end = time.time()
        elapsed = end - start

    out = model.scratch.output_conv(path_1)

    return elapsed

def embed(model, x):
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

def encoder(model, x):
    if torch.cuda.is_available():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model.pretrained.model.forward_flex(x)
        end.record()
    
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
    else:
        start = time.time()
        model.pretrained.model.forward_flex(x)
        end = time.time()
        elapsed = end - start
    
    return elapsed

def forward(model, x):
    if torch.cuda.is_available():
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        model.forward(x)
        end.record()
    
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
    else:
        start = time.time()
        model.forward(x)
        end = time.time()
        elapsed = end - start
    
    return elapsed

        
# warm up devices
for _ in range(5):
    model.forward(sample)

with torch.no_grad():
    for r in range(runs):
        timings['embed'][r] = embed(vit_rn, sample)
        timings['encoder'][r] = encoder(model, sample)
        timings['decoder'][r] = decoder(model, sample)
        timings['fusion'][r] = fusion(model, sample)
        timings['total'][r] = forward(model, sample)

# torch times in milliseconds, convert to seconds
if torch.cuda.is_available():
    for module in timings:
        timings[module] = timings[module] / 1000

if k8s:
    os.makedirs(os.path.dirname(script_output), exist_ok=True)
    df = pd.DataFrame({module: [times.mean(), times.std()] for module, times in timings.items()},
                      index=['mean','std'])
    df.to_csv(script_output)

for module, times in timings.items():
    print(f'{module}')
    print('-'*10)
    print('Mean times:', round(times.mean(),4))
    print('Std:', round(times.std(),4))
    print('-'*10)

