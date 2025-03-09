import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    """Detection Head for Faster R-CNN.
    
    This module performs the final classification and bounding box regression on the
    region proposals. It consists of two fully connected layers followed by separate
    branches for classification and bounding box regression.
    
    Args:
        in_channels (int): Number of input channels. Default: 256
        pooled_height (int): Height of the pooled feature maps. Default: 3
        pooled_width (int): Width of the pooled feature maps. Default: 3
        fc_dim (int): Dimension of the fully connected layers. Default: 1024
        num_classes (int): Number of object classes (excluding background). Default: 1
    """
    def __init__(self, in_channels=256, pooled_height=3, pooled_width=3, fc_dim=1024, num_classes=1):
        super(DetectionHead, self).__init__()
        
        self.flatten_dim = in_channels * pooled_height * pooled_width

        self.fc1 = nn.Linear(self.flatten_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

        # Classification branch (+1 for background class)
        self.cls_score = nn.Linear(fc_dim, num_classes + 1)

        # Bounding box regression branch (4 coordinates per class)
        self.bbox_pred = nn.Linear(fc_dim, 4 * num_classes)
    
    def forward(self, x):
        """Forward pass of the Detection Head.
        
        Args:
            x (torch.Tensor): Input tensor of pooled features from RoI pooling layer
                            Shape: (N, C, pooled_height, pooled_width)
        
        Returns:
            tuple:
                - cls_score (torch.Tensor): Classification scores for each RoI
                  Shape: (N, num_classes + 1)
                - bbox_pred (torch.Tensor): Bounding box regression deltas
                  Shape: (N, num_classes * 4)
        """
        x = x.view(-1, self.flatten_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred

