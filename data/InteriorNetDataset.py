import os
import re
import itertools
import numpy as np
from torch.utils.data import Dataset

from video_inference_common.video_inference.datasets import interiornet

def getlines(files: [str], subsample):
    ''' helper function to get stripped lines from multiple files'''
    names = []
    if subsample:
        for f in files:
            names.append(open(f).readline().strip())
            break
    else:
        for f in files:
            names.append(map(lambda x: x.strip(), open(f).readlines()))
        return list(itertools.chain.from_iterable(names))
            
    return names
    
class InteriorNetDataset(Dataset):
    
    def __init__(self, dataset_path: str, split: str='train',
                 n_folds: int=5, fold_idx: int=0, transform=None, 
                 subsample=False, no_folds=False):
        '''
        dataset_path: path to the folder containing the txts that specify dataset
                      (relative to ./dynamic-inference)
                      
        split: [train|val|test|nosplit] use the appropriate split
        n_folds: number of folds to separate data
        fold_idx: fold index to reserve for validation, [0, n_folds-1]
        transform: optional transform to be applied per sample
        subsample: take a subsample of all the training data
        '''
        
        assert 0 <= fold_idx <= n_folds
        
        self.transform = transform
        self.path = dataset_path
        self.subsample = subsample
        
        subsets = re.compile(f'.*?({"train" if split in ["train", "val"] else "test"}).*?')
        video_names = map(lambda p: os.path.join(dataset_path, p), 
                          filter(subsets.match, os.listdir(dataset_path)))
        
        self.videos = np.array(getlines(video_names, subsample))
        
        # not robust to irregular batch sizes
        if not subsample:
            fold_size, mod = divmod(len(self.videos), n_folds)

            assert mod == 0 # I'm sure there's a better way of handling this but I want experiments
            if split == 'train':
                self.videos = np.concatenate([self.videos[:fold_idx*fold_size], 
                                              self.videos[(fold_idx+1)*fold_size:]])
            if split == 'val':
                self.videos = self.videos[fold_idx*fold_size:(fold_idx+1)*fold_size]
        
    def __len__(self):
        # each video is 1000 frames
        return 100 * len(self.videos) if self.subsample else 1000 * len(self.videos) 
    
    def __getitem__(self, idx):
        
        # idx will come as video_index * frame_index
        if self.subsample:
            img_name = self.videos[0]
            frame_idx = idx
        else:
            img_name = self.videos[idx // 1000]
            frame_idx = idx % 1000
        
        im = interiornet.read_rgb(img_name, frame_idx)
        depth = interiornet.read_depth(img_name, frame_idx)
        
        if self.transform:
            im = self.transform({'image': im})['image']
        
        im = im / 255
        
        return {'image': im, 'depth': depth}