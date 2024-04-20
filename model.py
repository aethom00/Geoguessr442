import torch.nn as nn
import torch

class final_model(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super(final_model, self).__init__()
        self.fc1 = nn.Linear(in_features=in_features, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=2)

    def forward(self, x):
        x1 = self.fc1(x)
        x1 = nn.functional.relu(x1)
        x1 = self.fc2(x1)
        x1 = nn.functional.relu(x1)
        x1 = self.fc3(x1)
        return x1
    
    

    

        