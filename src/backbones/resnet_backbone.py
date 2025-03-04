import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import ResNet101_Weights

class ResNetBackbone(nn.Module):
    def __init__(self, output_layer_map):
        """
        output_layer_map: A dict that maps user-specified names -> actual submodule names
                          e.g. {
                               "layer1_out": "layer1",
                               "layer2_out": "layer2",
                               "layer3_out": "layer3",
                               "layer4_out": "layer4"
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
        # Store the outputs in a dict with the specified key
        self.features[key] = outputs

    def register_forward_hooks(self):
        # For each item in output_layer_map, register a forward hook
        for user_key, submodule_name in self.output_layer_map.items():
            # e.g., submodule_name = "layer2"
            submodule = getattr(self.resnet, submodule_name)
            # Register a hook that saves the output using user_key
            submodule.register_forward_hook(
                lambda module, inp, out, name=user_key: self.hook_fn(out, name)
            )

    def forward(self, x):
        # Clear previous features
        self.features = {}
        # Forward pass through the entire ResNet
        _ = self.resnet(x)
        # self.features now contains all hooked outputs
        return self.features
