from PIL import Image
from torchvision import transforms
from src.backbone import Backbone
from src.fpn import FPN
from src.rpn import RPN
from src.head import DetectionHead
import torch
import datasets

from src.utils import *

ds = datasets.load_dataset("rishitdagli/cppe-5")
train_ds = ds['train']
val_ds = ds['test']

mapped_train_ds = train_ds.map(filter_bboxes_in_sample)
filtered_train_ds = mapped_train_ds.filter(lambda sample: len(sample["objects"]["bbox"]) > 0)

mapped_val_ds = val_ds.map(filter_bboxes_in_sample)
filtered_val_ds = mapped_val_ds.filter(lambda sample: len(sample["objects"]["bbox"]) > 0)

import torch.optim as optim

output_layer_map = {
    'conv3': 16,
    'conv4': 23,
    'conv5': 30
}

layer_size_map = {
    'conv3': (75, 75),
    'conv4': (37, 37),
    'conv5': (18, 18)
}

C = 256
pooled_height, pooled_width = 7, 7
num_classes = len(ds['train'].features['objects'].feature['category'].names)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
backbone = Backbone(output_layer_map).to(device)
fpn = FPN().to(device)
rpn = RPN().to(device)
head = DetectionHead(in_channels=C, pooled_height=pooled_height, pooled_width=pooled_width, num_classes=num_classes).to(device)

optimizer = optim.Adam(
    list(backbone.parameters()) +
    list(fpn.parameters()) +
    list(rpn.parameters()) +
    list(head.parameters()),
    lr=1e-4
)

from tqdm import tqdm
import logging
from tqdm import tqdm
from datetime import datetime
import os

timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
# train_log_filename = f"train_metrics_{timestamp}.log"
train_log_filename = "train_metrics.log"
# val_log_filename = f"val_metrics_{timestamp}.log"
val_log_filename = "val_metrics.log"
log_dir = 'logs'

if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, train_log_filename),  # Log file name
    filemode='w',                  # Append mode; use 'w' to overwrite each time
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

train_logger = logging.getLogger("train")
val_logger = logging.getLogger("validation")
val_logger.setLevel(logging.INFO)
if val_logger.hasHandlers():
    val_logger.handlers.clear()
val_file_handler = logging.FileHandler(os.path.join(log_dir, val_log_filename), mode="w")
val_file_handler.setLevel(logging.INFO)
val_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
val_file_handler.setFormatter(val_formatter)
val_logger.addHandler(val_file_handler)
val_logger.propagate = False  # Prevent propagation to root logger

# torch.autograd.set_detect_anomaly(True)

checkpoint_dir = 'checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

train_dataset = DetectionDataset(filtered_train_ds, transform_pipeline, preprocess)
train_dataloader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=4, shuffle=True)
val_dataset = DetectionDataset(filtered_val_ds, transform_pipeline, preprocess)
val_dataloader = DataLoader(val_dataset, collate_fn=collate_fn, batch_size=4, shuffle=False)

img_shape = (600, 600)
anchors = generate_anchors()

layer_to_shifted_anchors = dict()
for k in layer_size_map.keys():
    layer_h, layer_w = layer_size_map[k]
    layer_to_shifted_anchors[k] = torch.from_numpy(shift((anchors), layer_h, layer_w, img_shape[0] // layer_h)).to(device).float()

num_epochs = 10
for epoch in range(num_epochs):
    train_loop(1, train_dataloader, backbone, fpn, rpn, head, optimizer, device,
               layer_to_shifted_anchors, img_shape, num_classes, pooled_height, pooled_width, train_logger)
    print(f'Starting Validation for epoch {epoch+1}')
    validation_loop(val_dataloader, backbone, fpn, rpn, head, device,
                    layer_to_shifted_anchors, img_shape, num_classes, pooled_height, pooled_width, val_logger)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
    torch.save({
        'epoch': epoch + 1,
        'backbone_state_dict': backbone.state_dict(),
        'fpn_state_dict': fpn.state_dict(),
        'rpn_state_dict': rpn.state_dict(),
        'head_state_dict': head.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Saved checkpoint for epoch {epoch+1} at {checkpoint_path}")