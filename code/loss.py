import torch
import torch.nn as nn


class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.margin = 1

    def forward(self, emb1, emb2, label):
        d = torch.norm(emb1 - emb2, p=2, dim=1)
        positive  = label.float()       * d.pow(2)
        negtive  =  (1 - label.float()) * torch.clamp(self.margin - d, min=0).pow(2)
        return (positive + negtive).mean()


class TripletLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.margin = 0.2

    def forward(self, anchor, positive, negative):
        d_pos = torch.norm(anchor - positive, p=2, dim=1)
        d_neg = torch.norm(anchor - negative,  p=2, dim=1)
        loss  =  torch.clamp(d_pos - d_neg + self.margin, min=0)
        return loss.mean()
