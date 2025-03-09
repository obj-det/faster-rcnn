"""
Backbone network implementation for Faster R-CNN using VGG16.

This module provides a feature extraction backbone based on VGG16 pre-trained on ImageNet.
It allows extraction of features from specific layers of the network through forward hooks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class Backbone(nn.Module):
    """VGG16-based backbone network for feature extraction.
    
    This backbone uses a pre-trained VGG16 network and extracts features from specified
    intermediate layers using forward hooks. The features are collected in a dictionary
    with keys corresponding to the layer names.
    
    Args:
        output_layer_map (dict): Mapping of layer names to their indices in the VGG16 
                               features module. For example: {'conv4_3': 22, 'conv5_3': 29}
    """
    def __init__(self, output_layer_map):
        super(Backbone, self).__init__()
        self.vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # self.vgg16.eval()
        self.output_layer_map = output_layer_map
        self.features = dict()
        self.register_forward_hooks()
    
    def hook_fn(self, outputs, key):
        """Forward hook function to store intermediate layer outputs.
        
        Args:
            outputs (torch.Tensor): Output tensor from the layer
            key (str): Layer name to use as key in the features dictionary
        """
        self.features[key] = outputs

    def register_forward_hooks(self):
        """Register forward hooks on the specified VGG16 layers.
        
        This method sets up hooks that will capture the output of specified layers
        during the forward pass. The outputs are stored in the features dictionary
        with the corresponding layer names as keys.
        """
        for layer_name, layer_idx in self.output_layer_map.items():
            self.vgg16[layer_idx].register_forward_hook(
                lambda _module, _inputs, outputs, name=layer_name: self.hook_fn(outputs, name)
            )

    def forward(self, x):
        """Forward pass through the backbone network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        
        Returns:
            dict: Dictionary containing the feature maps from specified layers.
                 Keys are layer names and values are the corresponding feature tensors.
        """
        self.features = dict()
        self.vgg16(x)
        return self.features