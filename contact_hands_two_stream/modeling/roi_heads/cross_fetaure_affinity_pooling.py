import torch
from torch import nn
from torch.nn import functional as F

class CrossFeatureAffinityPooling(nn.Module):

    def __init__(self, in_channels):
        super(CrossFeatureAffinityPooling, self).__init__()
        self.W_H = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)
        self.W_U = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0)

        self.bn_layer = nn.GroupNorm(32, in_channels)
        nn.init.constant_(self.bn_layer.weight, 0)
        nn.init.constant_(self.bn_layer.bias, 0)
    
    def forward(self, Hand, U):

        B, C, H, W = Hand.size()

        A = torch.matmul(
            self.W_H(Hand).view(B, C, H*W).permute(0, 2, 1),
            self.W_U(U).view(B, C, H*W)
        ) #[B, HW, HW]

        A = F.softmax(A, dim=-1)
        U = U.view(B, C, H*W).permute(0, 2, 1) #[B, HW, C]

        out = torch.matmul(A, U).permute(0, 2, 1).contiguous().view(B, C, H, W) #[B, HW, C]
        out = self.bn_layer(out) + Hand
        
        return out