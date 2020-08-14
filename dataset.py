
import numpy as np
import torch

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None, transforms=None):
        self.X = data
        self.y = labels
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        x = self.X.iloc[i, :]
        x = np.asarray(x).astype(np.float32)

        if self.transforms:
            x = self.transforms(x)
            
        if self.y is not None:
            return (x, self.y[i])
        else:
            return x