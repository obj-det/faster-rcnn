import torch
import torch.nn as nn
import torch.nn.functional as F

class DetectionHead(nn.Module):
    def __init__(self, in_channels=256, pooled_height=3, pooled_width=3, fc_dim=1024, num_classes=21):
        super(DetectionHead, self).__init__()
        
        self.flatten_dim = in_channels * pooled_height * pooled_width

        self.fc1 = nn.Linear(self.flatten_dim, fc_dim)
        self.fc2 = nn.Linear(fc_dim, fc_dim)

        self.cls_score = nn.Linear(fc_dim, num_classes + 1)

        self.bbox_pred = nn.Linear(fc_dim, 4 * num_classes)
    
    def forward(self, x):
        x = x.view(-1, self.flatten_dim)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        cls_score = self.cls_score(x)
        bbox_pred = self.bbox_pred(x)

        return cls_score, bbox_pred

