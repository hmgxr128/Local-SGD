import torch
import torch.nn as nn
from math import log
from torch.nn.functional import nll_loss


class LabelNoiseLoss(nn.Module):
    def __init__(self, num_classes=10, p=0.1, dim=-1):
        super(LabelNoiseLoss, self).__init__()
        self.p = p
        self.C = num_classes
        self.dim = dim
        self.minloss = -(1-p)*log(1-p) - p*log(p/(num_classes-1)) if p > 0 else 0
        

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=self.dim)
        if self.p == 0:
            # nll_loss(input, target), input is expected to be log probabilities
            loss = nll_loss(pred, target)
            return loss
        else:
            with torch.no_grad():
                smooth_dist = torch.zeros_like(pred)
                smooth_dist.fill_(self.p / (self.C - 1))
                smooth_dist.scatter_(1, target.data.unsqueeze(1), 1-self.p)
                noisy_target = torch.multinomial(smooth_dist, 1).squeeze()
            smooth_loss = torch.mean(torch.sum(- smooth_dist * pred, dim=self.dim))
            smooth_loss = smooth_loss - self.minloss
            # the first is the loss used for backward, the decond is the expected loss

            return nll_loss(pred, noisy_target)