import torch
import torch.nn as nn
import torch.nn.functional as F

class TitanicModel(nn.Module):
    '''The model for the titanic predictions, features listed soon'''
    def __init__(self):
        '''Sets up the model features'''
        fc1 = nn.Linear()
        fc2 = nn.Linear()
        fc3 = nn.Linear()
    def forward(self):
        pass