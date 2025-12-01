from torch.utils.data import Dataset
from collections import namedtuple

Batch = namedtuple("Batch", ["x", "cond", "mask"])

class SimpleTrajectoryDataset(Dataset):
    def __init__(self, x_tensor, cond_tensor, cond_mask):
        self.x_tensor     = x_tensor.float()        # [B, T, D]
        self.cond_tensor  = cond_tensor.float()     # [B, T, D]
        self.cond_mask    = cond_mask.float()       # [B, D, T]  ← 그대로 둠

    def __len__(self):
        return self.x_tensor.shape[0]

    def __getitem__(self, idx):
        return Batch(
            self.x_tensor[idx], 
            self.cond_tensor[idx], 
            self.cond_mask[idx]
        )