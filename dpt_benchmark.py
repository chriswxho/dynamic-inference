#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import glob
import torch
import torch.nn as nn
import cv2
import numpy as np
import time

import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet


# k8s paths
k8s = True
kube_repo = r'opt/repo/dynamic-inference'
kube_pvc = r'christh9-pvc'

# path settings
input_path = 'input'
output_path = 'output_monodepth'
model_path = 'weights/dpt_hybrid_nyu-2ce69ec7.pt'

if kube:
    input_path = os.path.join(kube_repo, input_path)
    output_path = os.path.join(kube_repo, output_path)
    model_path = os.path.join(kube_pvc, 'dpt-hybrid-nyu.pt')

model_type = 'dpt_hybrid'
optimize = True

runs = 500
timings = np.zeros((runs,2))


# select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device: %s" % device)

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


model.eval()

if optimize == True and device == torch.device("cuda"):
    model = model.to(memory_format=torch.channels_last)
    model = model.half()

model.to(device)


# get input
img_names = glob.glob(os.path.join(input_path, "*"))
num_images = len(img_names)

# create output folder
os.makedirs(output_path, exist_ok=True)


kitti_crop = False
absolute_depth = False

# input
img = np.random.rand(480,640,3)
if kitti_crop is True:
    height, width, _ = img.shape
    top = height - 352
    left = (width - 1216) // 2
    img = img[top : top + 352, left : left + 1216, :]

img_input = transform({"image": img})["image"]

sample = torch.from_numpy(img_input).to(device).unsqueeze(0)

if optimize == True and device == torch.device("cuda"):
    sample = sample.to(memory_format=torch.channels_last)
    sample = sample.half()

start, end = None, None
if device == torch.device('cuda'):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


# In[8]:


# custom decoder step for CUDA timing
def decoder(model, x):
    
    start, end = None, None
    if torch.cuda.is_available(): # assumes availability = in use
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    
    b, c, h, w = x.shape

    if torch.cuda.is_available():
        start.record()
    else:
        start = time.time()
        
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
    
    elapsed = None
    
    if torch.cuda.is_available():
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end)
    else:
        end = time.time()
        elapsed = end - start

    return out, elapsed


# In[9]:


# warm up devices
for _ in range(5):
    _ = model.forward(sample)

# compute
with torch.no_grad():
    for r in range(runs):
        # time encoder
        if device == torch.device('cuda'):
            start.record()
            _ = model.pretrained.model.forward_flex(sample)
            end.record()
            
            torch.cuda.synchronize()
            elapsed = start.elapsed_time(end)
            timings[r,0] = elapsed
            
        else:
            start = time.time()
            _ = model.pretrained.model.forward_flex(sample)
            end = time.time()
            timings[r,0] = end - start
            
        # time decoder
        _, elapsed = decoder(model, sample)
        timings[r,1] = elapsed
        
print('Mean times:', timings[:,0].mean(), timings[:,1].mean())
print('Std:', timings[:,0].std(), timings[:,1].std())

