from turtle import forward
import torch
import torchvision.models as models
import torch.nn as nn



class Resnet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = models.resnet18()
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features,10)
    
    def forward(self, x):
        out = self.model.forward(x)
        return out
