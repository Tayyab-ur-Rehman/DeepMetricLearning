import os
import random
import torch
from torchvision.models import resnet50
import torch.nn as nn
import torch.nn.functional as F


class DeepMetricLearning(nn.Module):
    def __init__(self,output_dim,train4thlayer=False):
        super().__init__()
        resnet = resnet50()
        resnet.load_state_dict(torch.load('weights/resnet50.pt', map_location='cpu'))
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in self.backbone.parameters():# freeze all
            param.requires_grad = False
        self.projection = nn.Sequential(nn.Linear(2048, 512), nn.ReLU(), nn.Linear(512, output_dim))
        
    def project(self, feat):  # only use feature and do forward pass for 2 layers only
        return F.normalize(self.projection(feat), p=2, dim=1)

    def forward(self,x):
        features = self.backbone(x)
        features = features.flatten(1)
        output = self.projection(features)
        output = F.normalize(output, p=2, dim=1)
        return output

        
model = DeepMetricLearning(output_dim=128, train4thlayer=True)