
import numpy as np
import torch

class CSVDataset(torch.utils.data.Dataset):
    def __init__(self, data, labels=None, transforms=None, device="cpu" ):
        self.X = torch.as_tensor( data.values, dtype=torch.float32, device=device )
        if labels is not None :
            self.y = torch.tensor( labels, device=device )
        else :
            self.y = None
        self.transforms = transforms
         
    def __len__(self):
        return (len(self.X))
    
    def __getitem__(self, i):
        x = self.X[i, :]

        if self.transforms:
            x = self.transforms(x)
            
        if self.y is not None:
            return (x, self.y[i])
        else:
            return x