import torch
import torch.nn as nn
import torch.nn.functional as F

class TitanicModel(nn.Module):
    '''The model for the titanic predictions, features listed soon'''
    def __init__(self):
        '''Sets up the model features'''
        super().__init__()
        # 109 nodes to input data
        self.fc1 = nn.Linear(108, 32)
        self.fc2 = nn.Linear(32, 8)
        self.fc3 = nn.Linear(8, 2)
        # 2 categories (survived or dead)

    def forward(self, X : torch.Tensor):
        X = F.relu(self.fc1(X))
        X = F.relu(self.fc2(X))
        X = F.relu(self.fc3(X))
        return F.softmax(X, dim=1)