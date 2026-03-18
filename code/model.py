import torchvision.models as resnet50
import torch.nn as nn
import torch.nn.functional as F

train4thlayer=1

class DeepMetricLearning(nn.Module):
    def __init__(self,output_dim,train4thlayer=False):
        super(self).__init__()
        resnet = resnet50(weights="IMAGENET1K_V1")
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        
        for param in self.backbone.parameters():# freeze all
            param.requires_grad = False
        if train4thlayer:
            for param in self.backbone.layer4.parameters(): # if we want to train last conv layer
                param.requires_grad = True

        self.projection = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Linear(512, output_dim)
        )
    def forward(self,x):
        features = self.backbone(x)
        features = features.flatten(1)
        output = self.projection(features)
        output = F.normalize(output, p=2, dim=1)
        return output

        
model = DeepMetricLearning(output_dim=128, train4thlayer=True)