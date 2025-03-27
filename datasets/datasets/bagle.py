import os
import torch
import numpy as np
import torch.utils.data as data
from .io import IO
from .build import DATASETS
from utils.logger import *

@DATASETS.register_module()
class bagle(data.Dataset):
    def __init__(self, data_path):
        self.data_root = data_path
        self.npoints = 16384
        
        self.file_list = []
        for filename in os.listdir(self.data_root):
            if filename.endswith('.txt'):
                model_id = filename.split('.')[0]
                self.file_list.append({
                    'taxonomy_id': 'bagel',  # 类别信息设为未知
                    'model_id': model_id,
                    'file_path': os.path.join(self.data_root, filename)
                })

        print_log(f'[DATASET] {len(self.file_list)} instances were loaded', logger='BagleDataset')

        self.permutation = np.arange(self.npoints)
        
        
    def pc_norm(self, pc):
        """ pc: NxC, return NxC """
        centroid = np.mean(pc, axis=0)
        pc = pc - centroid
        m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
        pc = pc / m
        return pc
        

    def random_sample(self, pc, num):
        if num >= len(pc):
            return pc
        else:
            indices = np.random.permutation(len(pc))[:num]
            return pc[indices]
        
    def __getitem__(self, idx):
        sample = self.file_list[idx]

        data = IO.get(sample['file_path']).astype(np.float32)

        data = self.random_sample(data, self.npoints)
        data = self.pc_norm(data)
        data = torch.from_numpy(data).float()
        return sample['taxonomy_id'], sample['model_id'], data

    def __len__(self):
        return len(self.file_list)