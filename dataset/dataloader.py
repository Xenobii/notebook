import h5py
import torch
import numpy as np
from torch.utils.data import Dataset



class MaestroDataset(Dataset):
    def __init__(self, f_h5, split=None):
        self.f_h5  = f_h5
        self.split = split

        with h5py.File(f_h5, "r") as h5:
            self.keys = [
                k for k in h5.keys()
                if split is None or h5[k].attrs["split"] == split
            ]

    def __len__(self):
        return len(self.keys)
    
    def __getitem__(self, idx):
        key = self.keys[idx]

        with h5py.File(self.f_h5, "r") as h5:
            group = h5[key]
            wave = torch.tensor(np.array(group["wav"]), dtype=torch.float32)
            meta = dict(group.attrs)
        
        return wave, meta