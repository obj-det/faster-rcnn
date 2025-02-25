import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    def __init__(self, in_channels=256, num_anchors=9):
        super(RPN, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv_cls = nn.Conv2d(256, num_anchors * 2, kernel_size=1)
        self.conv_reg = nn.Conv2d(256, num_anchors * 4, kernel_size=1)

    
    def forward(self, x):
        x = F.relu(self.initial_conv(x))
        cls = self.conv_cls(x)
        reg = self.conv_reg(x)
        return cls, reg