import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    """Feature Pyramid Network (FPN) implementation.
    
    FPN enhances a convolutional network for object detection by adding a top-down pathway 
    and lateral connections to create a feature pyramid that combines low-resolution, 
    semantically strong features with high-resolution, semantically weak features.
    
    Args:
        out_channels (int): Number of output channels for each FPN level. Default: 256
        in_channels (list): List of input channels for each input feature map. Default: [256, 512, 512]
    """
    def __init__(self, out_channels=256, in_channels=[256, 512, 512]):
        super(FPN, self).__init__()
        self.lat_convs = nn.ModuleList([nn.Conv2d(in_channels=c, out_channels=out_channels, kernel_size=1) for c in in_channels])
        self.smooth_convs = nn.ModuleList([nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1) for _ in range(len(in_channels)-1)])

    def forward(self, features_map):
        """Forward pass of the FPN.
        
        Args:
            features_map (dict): Dictionary of feature maps from the backbone network,
                               with keys representing the level and values being the feature tensors.
        
        Returns:
            dict: Dictionary of processed feature maps at different pyramid levels.
                 Each level contains features of the same channel dimension (out_channels).
        """
        output = dict()
        keys, features = zip(*sorted(features_map.items(), key=lambda x: x[0]))
        aligned_features = [self.lat_convs[i](features[i]) for i in range(len(features))]
        output[keys[-1]] = aligned_features[-1]
        for i in range(len(aligned_features)-2, -1, -1):
            aligned_features[i] = F.interpolate(aligned_features[i+1], size=aligned_features[i].shape[-2:], mode='nearest') + aligned_features[i]
            aligned_features[i] = self.smooth_convs[i](aligned_features[i])
            output[keys[i]] = aligned_features[i]
        return output

        