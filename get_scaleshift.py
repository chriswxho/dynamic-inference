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
from tqdm import tqdm
import argparse

import torch
import torch.nn.functional as F
from torchvision.transforms import Compose

import pytorch_lightning as pl

from dpt.plmodels import InteriorNetDPT
from dpt.transforms import Resize, NormalizeImage, PrepareForNet
from data.InteriorNetDataset import InteriorNetDataset

torch.manual_seed(0)
np.random.seed(0)


def compute_scale_and_shift(disp):
    mask = ~torch.isnan(disp)
    shift = torch.median(disp[mask])
    scale = (disp[mask] - shift).abs().mean()
    return scale, shift
    
def get_scale_and_shift(args):
    
    # k8s paths
    k8s_repo = r'opt/repo/dynamic-inference'
    k8s_pvc = r'christh9-pvc'

    # path settings
    input_path = 'input'
    output_path = 'output_monodepth'
    model_path = 'weights/dpt_hybrid_nyu-2ce69ec7.pt'
    dataset_path = 'video_inference_common/resources'
    logs_path = 'train-logs'

    if args['k8s']:
        input_path = os.path.join(k8s_repo, input_path)
        output_path = os.path.join(k8s_repo, output_path)
        model_path = os.path.join(k8s_pvc, 'dpt-hybrid-nyu.pt')
        dataset_path = os.path.join(k8s_repo, dataset_path)
        logs_path = os.path.join(k8s_pvc, logs_path)
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
    
    # model setup
    model = InteriorNetDPT(1, 1e-4, 1, model_path, s=0.000305, t=0.1378)
    model.freeze()
    model.cuda()
    
    # dataset setup
    dataset = InteriorNetDataset(dataset_path, split='train', transform=transform, no_folds=True)
    
    assert 0 < args['ratio'] <= 1
    
    idxs = np.random.choice(len(dataset), size=round(len(dataset) * args['ratio']), replace=False)

    print(f'Created datasets and models in {timedelta(seconds=round(time.time()-start,2))}')
    
    scale = []
    shift = []
    
    with torch.no_grad():
        for idx in tqdm(idxs, ncols=40):
            batch = dataset[idx]
            im, depth = torch.from_numpy(batch['image']).unsqueeze(0).cuda(), torch.from_numpy(batch['depth']).cuda()
            sd, td = compute_scale_and_shift(depth)
            sp, tp = compute_scale_and_shift(model(im).squeeze())
            scale.append(F.relu(sd/sp).item())
            shift.append((td - tp * scale[-1]).item())
        
    scale = torch.tensor(scale).mean().item()
    shift = torch.tensor(shift).mean().item()
    
    print(f's: {scale}')
    print(f't: {shift}')
    print(f"ratio used: {round(args['ratio']*100)}%")
    
    df = pd.DataFrame({ 'scale': [scale], 
                        'shift': [shift] })
    
    dest_path = os.path.join(logs_path, f"st_{args['ratio']}") 
    print(f'Results saved in {dest_path}')
    df.to_csv(dest_path)
    

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Get the scale/shift of the DPT and the dataset.')
    
    parser.add_argument(
        '-k', '--k8s', action='store_true'
    )
    
    parser.add_argument(
        '-r', '--ratio', help='proportion of training data to calculate stats on', default=1, type=float
    )
    
    args = dict(vars(parser.parse_args()))

    get_scale_and_shift(args)