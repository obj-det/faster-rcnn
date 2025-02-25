import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import VGG16_Weights

class Backbone(nn.Module):
    def __init__(self, output_layer_map):
        super(Backbone, self).__init__()
        self.vgg16 = models.vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        # self.vgg16.eval()
        self.output_layer_map = output_layer_map
        self.features = dict()
        self.register_forward_hooks()
    
    def hook_fn(self, outputs, key):
        self.features[key] = outputs

    def register_forward_hooks(self):
        for layer_name, layer_idx in self.output_layer_map.items():
            print(layer_name, layer_idx)
            self.vgg16[layer_idx].register_forward_hook(lambda _module, _inputs, outputs, name=layer_name: self.hook_fn(outputs, name))

    def forward(self, x):
        self.features = dict()
        self.vgg16(x)
        return self.features