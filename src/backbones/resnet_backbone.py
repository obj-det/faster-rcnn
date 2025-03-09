"""
ResNet-101 backbone network implementation for Faster R-CNN.

This module provides a feature extraction backbone based on ResNet-101 pre-trained on ImageNet.
It allows extraction of features from specific layers of the network through forward hooks,
making it suitable for feature pyramid network (FPN) and other multi-scale architectures.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights

class ResNetBackbone(nn.Module):
    """ResNet-101 based backbone network for feature extraction.
    
    This backbone uses a pre-trained ResNet-101 network and extracts features from specified
    intermediate layers using forward hooks. The features are collected in a dictionary
    with keys corresponding to user-defined names. The final fully connected layer is 
    replaced with an identity function as it's not needed for feature extraction.
    
    Args:
        output_layer_map (dict): Mapping of user-defined names to ResNet layer names.
                                For example:
                                {
                                    "conv1": "layer1",  # Resolution 1/4
                                    "conv2": "layer2",  # Resolution 1/8
                                    "conv3": "layer3",  # Resolution 1/16
                                    "conv4": "layer4"   # Resolution 1/32
                                }
    """
    def __init__(self, output_layer_map):
        """
        output_layer_map: A dict that maps user-specified names -> actual submodule names
                          e.g. {
                               "conv1": "layer1",
                               "conv2": "layer2",
                               "conv3": "layer3",
                               "conv4": "layer4"
                          }
        """
        super(ResNetBackbone, self).__init__()
        # Load a pretrained resnet101
        self.resnet = models.resnet101(weights=ResNet101_Weights.IMAGENET1K_V1)
        # If you don't need the final fc, you can optionally remove it:
        self.resnet.fc = nn.Identity()

        self.output_layer_map = output_layer_map
        self.features = {}
        self.register_forward_hooks()

    def hook_fn(self, outputs, key):
        """Forward hook function to store intermediate layer outputs.
        
        Args:
            outputs (torch.Tensor): Output tensor from the layer
            key (str): User-defined key to use in the features dictionary
        """
        # Store the outputs in a dict with the specified key
        self.features[key] = outputs

    def register_forward_hooks(self):
        """Register forward hooks on the specified ResNet layers.
        
        This method sets up hooks that will capture the output of specified layers
        during the forward pass. For each layer specified in output_layer_map,
        it registers a hook that saves the layer's output using the corresponding
        user-defined key.
        """
        # For each item in output_layer_map, register a forward hook
        for user_key, submodule_name in self.output_layer_map.items():
            # e.g., submodule_name = "layer2"
            submodule = getattr(self.resnet, submodule_name)
            # Register a hook that saves the output using user_key
            submodule.register_forward_hook(
                lambda module, inp, out, name=user_key: self.hook_fn(out, name)
            )

    def forward(self, x):
        """Forward pass through the backbone network.
        
        Args:
            x (torch.Tensor): Input tensor of shape (N, C, H, W)
        
        Returns:
            dict: Dictionary containing the feature maps from specified layers.
                 Keys are the user-defined names from output_layer_map and
                 values are the corresponding feature tensors.
        """
        # Clear previous features
        self.features = {}
        # Forward pass through the entire ResNet
        _ = self.resnet(x)
        # self.features now contains all hooked outputs
        return self.features
