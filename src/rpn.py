import torch
import torch.nn as nn
import torch.nn.functional as F


class RPN(nn.Module):
    """Region Proposal Network (RPN) module.
    
    RPN is a fully convolutional network that simultaneously predicts object bounds and 
    objectness scores at each position. It takes feature maps as input and outputs a set 
    of rectangular object proposals, each with an objectness score.
    
    Args:
        in_channels (int): Number of input channels from the backbone/FPN. Default: 256
        num_anchors (int): Number of anchor boxes per spatial position. Default: 9
    """
    def __init__(self, in_channels=256, num_anchors=9):
        super(RPN, self).__init__()
        self.initial_conv = nn.Conv2d(in_channels, 256, kernel_size=3, padding=1)
        self.conv_cls = nn.Conv2d(256, num_anchors * 2, kernel_size=1)  # 2 classes: foreground and background
        self.conv_reg = nn.Conv2d(256, num_anchors * 4, kernel_size=1)  # 4 values: (tx, ty, tw, th) for each anchor

    def forward(self, x):
        """Forward pass of the RPN.
        
        Args:
            x (torch.Tensor): Input feature map of shape (N, C, H, W)
        
        Returns:
            tuple:
                - cls (torch.Tensor): Classification scores for each anchor
                  Shape: (N, num_anchors * 2, H, W)
                - reg (torch.Tensor): Regression values for each anchor
                  Shape: (N, num_anchors * 4, H, W)
        """
        x = F.relu(self.initial_conv(x))
        cls = self.conv_cls(x)
        reg = self.conv_reg(x)
        return cls, reg