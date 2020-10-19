from torch import nn
import torch 
import torch.nn.functional as F 

class SpatialAttention(nn.Module):

    def __init__(self, in_channels, num_attention_maps):
        super(SpatialAttention, self).__init__()
        self.in_channels = in_channels
        self.num_attention_maps = num_attention_maps
        self.W = nn.Conv2d(in_channels, num_attention_maps, 
        kernel_size=1, padding=0, bias=False
        )
        self.phi = nn.Conv2d(in_channels, self.num_attention_maps*4,
        kernel_size=1, padding=0, bias=False
        )

    def forward(self, x):

        B, C, H, W = x.size()
        
        spatial_attention_maps = F.softmax(
            self.W(x).view(B, self.num_attention_maps, H*W), dim=2
            ).unsqueeze(2) #[B, L, 1, HW]
        
        contact_scores = self.phi(x).view(B, self.num_attention_maps, 4, H*W) #[B, L, 4, HW]            
        contact_scores = torch.sum(
            spatial_attention_maps * contact_scores, dim=3, keepdim=False
        ) #[B, L, 4]

        contact_scores = torch.mean(contact_scores, dim=1, keepdim=False) #[B, 4]

        return contact_scores