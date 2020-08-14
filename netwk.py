import torch.nn as nn
import torch
 
 
class Net(nn.Module):
 
    def __init__(self, hidden=512):
        super(Net, self).__init__()
        self.transfer = torch.nn.CELU()
        self.final_transfer = torch.nn.GELU()

        self.fc1 = nn.Linear(6, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, hidden)
        self.fc6 = nn.Linear(hidden, 1)
        self.register_buffer('norm',
                             torch.tensor([200.0,
                                           198.0,
                                           200.0,
                                           0.4,
                                           0.2,
                                           0.2]))
 
    def forward(self, x):
        # normalize the parameter to range [0-1] 
        x = x / self.norm
        x = self.transfer(self.fc1(x))
        x = self.transfer(self.fc2(x))
        x = self.transfer(self.fc3(x))
        x = self.transfer(self.fc4(x))
        x = self.transfer(self.fc5(x))
        return self.fc6(x).squeeze()

