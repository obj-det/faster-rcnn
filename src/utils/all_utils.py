import torch
import numpy as np

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    
    Parameters:
        anchor (ndarray): A 4-element array (x1, y1, x2, y2).
        
    Returns:
        w, h, x_ctr, y_ctr: width, height, x center, and y center of the anchor.
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given vectors of widths (ws) and heights (hs) around a center (x_ctr, y_ctr),
    output a set of anchors (windows).
    
    Parameters:
        ws (ndarray): widths, shape (N,)
        hs (ndarray): heights, shape (N,)
        x_ctr (float): x center
        y_ctr (float): y center
        
    Returns:
        anchors (ndarray): Array of shape (N, 4)
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((
        x_ctr - 0.5 * (ws - 1),
        y_ctr - 0.5 * (hs - 1),
        x_ctr + 0.5 * (ws - 1),
        y_ctr + 0.5 * (hs - 1)
    ))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    
    Parameters:
        anchor (ndarray): Base anchor (4,)
        ratios (list): List of aspect ratios
        
    Returns:
        anchors (ndarray): Array of shape (len(ratios), 4)
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / np.array(ratios)
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * np.array(ratios))
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    
    Parameters:
        anchor (ndarray): Anchor (4,)
        scales (list): List of scales
        
    Returns:
        anchors (ndarray): Array of shape (len(scales), 4)
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * np.array(scales)
    hs = h * np.array(scales)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """
    Generate base anchors by enumerating aspect ratios and scales.
    
    Parameters:
        base_size (int): The size of the base anchor (typically the stride of the feature map).
        ratios (list): Aspect ratios (height/width) to enumerate.
        scales (list): Anchor scales (multiplicative factors).
        
    Returns:
        anchors (ndarray): Array of shape (N, 4), where N = len(ratios) * len(scales).
                           Each row is (x1, y1, x2, y2).
    """
    # Create a base anchor centered at (0, 0)
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])
    
    # Enumerate anchors for each aspect ratio
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    
    # For each ratio anchor, enumerate anchors for each scale
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    
    return anchors

def shift(anchors, feat_height, feat_width, feat_stride):
    shift_x = np.arange(0, feat_width * feat_stride, feat_stride)
    shift_y = np.arange(0, feat_height * feat_stride, feat_stride)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), 
        shift_y.ravel(),
        shift_x.ravel(), 
        shift_y.ravel()
    )).transpose()

    anchors = anchors.reshape(1, -1, 4) + shifts.reshape(-1, 1, 4)

    all_anchors = anchors.reshape((anchors.shape[0] * anchors.shape[1], 4))

    return all_anchors

def decode_boxes(anchors, deltas, im_shape):
    """
    Decode bounding boxes from anchors and deltas (both of shape [N, 4]) using PyTorch tensors.
    
    Parameters:
        anchors (Tensor): Tensor of shape [N, 4] with anchor boxes in [x1, y1, x2, y2] format.
        deltas (Tensor): Tensor of shape [N, 4] with predicted offsets (dx, dy, dw, dh).
    
    Returns:
        pred_boxes (Tensor): Tensor of shape [N, 4] with final bounding boxes.
    """
    # Compute widths, heights, and center coordinates for the anchors.
    widths  = anchors[:, 2] - anchors[:, 0] + 1.0  # Shape: [N]
    heights = anchors[:, 3] - anchors[:, 1] + 1.0    # Shape: [N]
    ctr_x   = anchors[:, 0] + 0.5 * (widths - 1.0)     # Shape: [N]
    ctr_y   = anchors[:, 1] + 0.5 * (heights - 1.0)    # Shape: [N]
    
    # Extract the deltas for each anchor.
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    
    # Apply the deltas: adjust center coordinates and sizes.
    pred_ctr_x = dx * widths + ctr_x    # New center x
    pred_ctr_y = dy * heights + ctr_y   # New center y
    pred_w = torch.exp(dw) * widths     # New width
    pred_h = torch.exp(dh) * heights    # New height
    
    # Convert center/size back to corner coordinates.
    x1 = torch.clamp(pred_ctr_x - 0.5 * (pred_w - 1.0), 0, im_shape[1] - 1)
    y1 = torch.clamp(pred_ctr_y - 0.5 * (pred_h - 1.0), 0, im_shape[0] - 1)
    x2 = torch.clamp(pred_ctr_x + 0.5 * (pred_w - 1.0), 0, im_shape[1] - 1)
    y2 = torch.clamp(pred_ctr_y + 0.5 * (pred_h - 1.0), 0, im_shape[0] - 1)
    pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # pred_boxes[:, 0] = pred_boxes[:, 0].clamp(0, im_shape[1] - 1)  # x1
    # pred_boxes[:, 1] = pred_boxes[:, 1].clamp(0, im_shape[0] - 1)  # y1
    # pred_boxes[:, 2] = pred_boxes[:, 2].clamp(0, im_shape[1] - 1)  # x2
    # pred_boxes[:, 3] = pred_boxes[:, 3].clamp(0, im_shape[0] - 1)  # y2
    
    return pred_boxes

def collect_rpn_deltas(rpn_deltas):
    B, C, H, W = rpn_deltas.shape
    num_anchors = C // 4
    rpn_deltas = rpn_deltas.view(B, num_anchors, 4, H, W)
    rpn_deltas = rpn_deltas.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4)
    batch_indices = torch.arange(B, dtype=rpn_deltas.dtype, device=rpn_deltas.device).view(B, 1)
    batch_indices = batch_indices.expand(B, rpn_deltas.size(1)).unsqueeze(2)
    rpn_deltas = torch.cat([batch_indices, rpn_deltas], dim=2).view(-1, 5)
    return rpn_deltas

def generate_rois(rpn_deltas, shifted_anchors, im_shape):
    """
    Generate candidate RoIs by applying predicted bounding-box deltas to the anchors.
    
    Parameters:
        feature_map (Tensor): Feature map from the backbone network.
        rpn_deltas (Tensor): Predicted bounding-box deltas from the RPN.
        anchors (Tensor): Anchor boxes in [x1, y1, x2, y2] format.
        im_shape (tuple): Shape of the input image (H x W).
        
    Returns:
        rois (Tensor): Region of interests (RoIs) in [x1, y1, x2, y2] format.
    """
    im_h, im_w = im_shape
    B, C, H, W = rpn_deltas.shape

    num_anchors = C // 4

    rpn_deltas = rpn_deltas.view(B, num_anchors, 4, H, W)

    rpn_deltas = rpn_deltas.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4)
    # anchors = shift(anchors, H, W, im_h // H)

    # anchors = torch.from_numpy(anchors).to(rpn_deltas.device).float()

    rois = torch.stack([decode_boxes(shifted_anchors, rpn_deltas[i], (im_h, im_w)) for i in range(B)], dim=0)

    batch_indices = torch.arange(B, dtype=rois.dtype, device=rois.device).view(B, 1)

    batch_indices = batch_indices.expand(B, rois.size(1)).unsqueeze(2)

    rois = torch.cat([batch_indices, rois], dim=2).view(-1, 5)

    return rois

import os
import yaml
import logging
from datetime import datetime
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from torchvision import transforms
import albumentations as A
from collections import defaultdict

from src.backbone import Backbone
from src.fpn import FPN
from src.rpn import RPN
from src.head import DetectionHead

from src.utils.anchor_utils import generate_anchors, shift

import datasets
from src.utils.data_utils import filter_bboxes_in_sample

def load_config(config_path):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_dataset(config):
    """Set up and return datasets based on configuration"""
    
    ds = datasets.load_dataset(config['dataset']['name'])
    train_ds = ds['train']
    val_ds = ds['val']

    tgt_categories = config['dataset']['tgt_categories']
    
    # mapped_train_ds = train_ds.map(lambda sample: filter_bboxes_in_sample(sample, tgt_categories), load_from_cache_file=False)
    # mapped_val_ds = val_ds.map(lambda sample: filter_bboxes_in_sample(sample, tgt_categories), load_from_cache_file=False)
    mapped_train_ds = train_ds.map(lambda sample: filter_bboxes_in_sample(sample, tgt_categories))
    mapped_val_ds = val_ds.map(lambda sample: filter_bboxes_in_sample(sample, tgt_categories))
    
    if config['dataset']['filter_empty_boxes']:
        # filtered_train_ds = mapped_train_ds.filter(lambda sample: len(sample["objects"]["bbox"]) > 0, load_from_cache_file=False)
        # filtered_val_ds = mapped_val_ds.filter(lambda sample: len(sample["objects"]["bbox"]) > 0, load_from_cache_file=False)
        filtered_train_ds = mapped_train_ds.filter(lambda sample: len(sample["objects"]["bbox"]) > 0)
        filtered_val_ds = mapped_val_ds.filter(lambda sample: len(sample["objects"]["bbox"]) > 0)
    else:
        filtered_train_ds = mapped_train_ds
        filtered_val_ds = mapped_val_ds

    filtered_train_ds[0]['image'].save('sample_train.jpg')
    filtered_val_ds[0]['image'].save('sample_val.jpg')

    # Count categories for determining num_classes
    category_to_count = defaultdict(int)
    for sample in filtered_train_ds:
        for obj in sample["objects"]["category"]:
            category_to_count[obj] += 1
    
    print(category_to_count)

    print(f"Found {len(category_to_count)} classes in dataset")
    return filtered_train_ds, filtered_val_ds, category_to_count

def setup_preprocess_transform(config):
    """Create a preprocessing transform that converts images to tensors and normalizes them."""
    preprocess_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['image']['mean'],
            std=config['image']['std']
        )
    ])
    return preprocess_pipeline

def setup_augmentation_transform(config, mode):
    """Create an augmentation pipeline with Albumentations, including bbox support."""
    aug_transforms = []

    if mode == 'train':
        # Apply horizontal flip if specified.
        if config['augmentation'].get('horizontal_flip_prob', 0) > 0:
            aug_transforms.append(A.HorizontalFlip(p=config['augmentation']['horizontal_flip_prob']))

        # Apply vertical flip if specified.
        if config['augmentation'].get('vertical_flip_prob', 0) > 0:
            aug_transforms.append(A.VerticalFlip(p=config['augmentation']['vertical_flip_prob']))

        # Use RandomResizedCrop instead of RandomCrop.
        # This crop will take a random portion of the image (e.g., between 80% and 100% of the original area)
        # and then resize it to the target shape.
        # if config['augmentation'].get('random_crop_prob', 0) > 0:
        #     height, width = config['image']['shape']
        #     aug_transforms.append(A.RandomResizedCrop(
        #         size=(height, width),
        #         scale=(0.8, 1.0),
        #         ratio=(0.75, 1.33),
        #         p=config['augmentation']['random_crop_prob']
        #     ))


        # Brightness and contrast adjustment.
        if config['augmentation'].get('brightness_range', []) or config['augmentation'].get('contrast_range', []):
            brightness_limit = (config['augmentation']['brightness_range'][0] - 1, config['augmentation']['brightness_range'][1] - 1)
            contrast_limit = (config['augmentation']['contrast_range'][0] - 1, config['augmentation']['contrast_range'][1] - 1)
            aug_transforms.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.5))

        # Saturation and hue adjustment.
        if config['augmentation'].get('hue_range', []) or config['augmentation'].get('saturation_range', []):
            aug_transforms.append(A.HueSaturationValue(
                hue_shift_limit=int(config['augmentation']['hue_range'][1]*100),
                sat_shift_limit=int((config['augmentation']['saturation_range'][1]-1)*100),
                val_shift_limit=0,
                p=0.5
            ))
    
    # Finally, if you want to ensure a consistent output size, you can also apply a Resize at the end.
    # However, RandomResizedCrop already outputs images at the specified size.
    height, width = config['image']['shape']
    aug_transforms.append(A.Resize(height, width))
    
    transform_pipeline = A.Compose(
        aug_transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"])
    )
    return transform_pipeline

def setup_models(config, num_classes, device, ckpt_path=None):
    """Initialize and return models based on configuration"""
    
    # Extract configuration
    backbone_config = config['model']['backbone']
    fpn_config = config['model']['fpn']
    
    # Prepare model parameters
    output_layer_map = backbone_config['output_layers']
    layer_depth_map = backbone_config['layer_depths']
    
    # Setup FPN parameters
    fpn_in_channels = [layer_depth_map[k] for k in sorted(output_layer_map.keys(), key=lambda x: int(x[-1]))]
    fpn_out_channels = fpn_config['out_channels']

    # Create models
    backbone = Backbone(output_layer_map).to(device)
    fpn = FPN(in_channels=fpn_in_channels, out_channels=fpn_out_channels).to(device)
    
    # Setup RPN parameters
    rpn_config = config['model']['rpn']
    num_anchors = len(rpn_config['anchor_box_ratios']) * len(rpn_config['anchor_box_scales'])
    rpn = RPN(in_channels=fpn_out_channels, num_anchors=num_anchors).to(device)
    
    # Setup Detection Head
    head_config = config['model']['head']
    head = DetectionHead(
        in_channels=fpn_out_channels,
        pooled_height=head_config['pooled_height'],
        pooled_width=head_config['pooled_width'],
        num_classes=num_classes
    ).to(device)
    
    if ckpt_path:
        ckpt = torch.load(ckpt_path, weights_only=False)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        fpn.load_state_dict(ckpt['fpn_state_dict'])
        rpn.load_state_dict(ckpt['rpn_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])

    return backbone, fpn, rpn, head

def setup_optimizer(config, models):
    """Set up optimizer based on configuration"""
    backbone, fpn, rpn, head = models
    
    if config['training']['optimizer'].lower() == 'adam':
        optimizer = optim.Adam(
            list(backbone.parameters()) +
            list(fpn.parameters()) +
            list(rpn.parameters()) +
            list(head.parameters()),
            lr=float(config['training']['learning_rate']),
            weight_decay=float(config['training']['weight_decay'])
        )
    elif config['training']['optimizer'].lower() == 'sgd':
        optimizer = optim.SGD(
            list(backbone.parameters()) +
            list(fpn.parameters()) +
            list(rpn.parameters()) +
            list(head.parameters()),
            lr=float(config['training']['learning_rate']),
            momentum=float(config['training']['momentum']),
            weight_decay=float(config['training']['weight_decay'])
        )
    else:
        raise ValueError(f"Unsupported optimizer: {config['training']['optimizer']}")
    
    return optimizer

def setup_logging(config):
    """Set up logging based on configuration"""
    log_dir = config['logging']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Get log filenames, use timestamp if not specified
    train_log_filename = config['logging'].get('train_log_filename')
    val_log_filename = config['logging'].get('val_log_filename')
    
    if not train_log_filename:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_filename = f"train_metrics_{timestamp}.log"
    
    if not val_log_filename:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        val_log_filename = f"val_metrics_{timestamp}.log"
    
    # Setup training logger
    logging.basicConfig(
        filename=os.path.join(log_dir, train_log_filename),
        filemode='w',
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    train_logger = logging.getLogger("train")
    
    # Setup validation logger
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
    
    return train_logger, val_logger

def setup_dataloaders(config, train_dataset, val_dataset, collate_fn):
    """Set up data loaders based on configuration"""
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=config['training']['batch_size'],
        shuffle=config['training']['shuffle'],
        collate_fn=collate_fn
    )
    
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=config['validation']['batch_size'],
        shuffle=config['validation']['shuffle'],
        collate_fn=collate_fn
    )
    
    return train_dataloader, val_dataloader

def setup_anchors(config, device):
    """Set up anchor boxes based on configuration"""
    
    rpn_config = config['model']['rpn']
    backbone_config = config['model']['backbone']
    img_shape = config['image']['shape']
    
    # Generate base anchors
    anchors = generate_anchors(
        ratios=rpn_config['anchor_box_ratios'],
        scales=rpn_config['anchor_box_scales']
    )
    
    # Shift anchors for each FPN level
    layer_to_shifted_anchors = {}
    for k, size in backbone_config['layer_sizes'].items():
        layer_h, layer_w = size
        layer_to_shifted_anchors[k] = torch.from_numpy(
            shift((anchors), layer_h, layer_w, img_shape[0] // layer_h)
        ).to(device).float()
    
    return layer_to_shifted_anchors

def setup_checkpoint_dir(config):
    """Set up checkpoint directory based on configuration"""
    checkpoint_dir = config['checkpoints']['save_dir']
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    return checkpoint_dir

from PIL import Image
from torchvision import transforms
import numpy as np
import torch

from torch.utils.data import Dataset, DataLoader
import albumentations as A
import cv2

class DetectionDataset(Dataset):
    def __init__(self, hf_dataset, albumentations_transform, preprocess_transform):
        """
        Args:
            hf_dataset: A Hugging Face dataset for cppe-5 object detection.
            albumentations_transform: An albumentations.Compose transform that
                applies data augmentation and expects keys "image", "bboxes", "category".
            preprocess_transform: A torchvision transform to preprocess images (e.g., for VGG16).
        """
        self.dataset = hf_dataset
        self.albumentations_transform = albumentations_transform
        self.preprocess_transform = preprocess_transform
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        sample = self.dataset[idx]

        image = np.array(sample["image"].convert("RGB"))

        bboxes = sample['objects']["bbox"]  # list of bounding boxes
        labels = sample['objects']["category"]  # list of integer labels

        transformed = self.albumentations_transform(image=image, bboxes=bboxes, category=labels)

        aug_image = transformed["image"]
        aug_bboxes = transformed["bboxes"]
        aug_labels = transformed["category"]

        # print(idx, aug_image.shape, len(aug_bboxes), len(aug_labels))

        image_pil = Image.fromarray(aug_image).convert("RGB")
        image_tensor = self.preprocess_transform(image_pil)

        boxes_tensor = torch.tensor(aug_bboxes, dtype=torch.float32)
        labels_tensor = torch.tensor(aug_labels, dtype=torch.int64)

        boxes_with_label = torch.cat([boxes_tensor, labels_tensor.unsqueeze(1)], dim=1)

        target = {
            "boxes": boxes_tensor,   # shape: [num_boxes, 4]
            "labels": labels_tensor,  # shape: [num_boxes]
            "boxes_with_label": boxes_with_label  # shape: [num_boxes, 5]
        }

        return image_tensor, target

def collate_fn(batch):
    # print(type(batch))
    images, targets = list(zip(*batch))
    images = torch.stack(images, dim=0)
    return images, targets

def filter_bboxes_in_sample(sample, tgt_categories):
    valid_bboxes = []
    valid_categories = []
    valid_ids = [] if "id" in sample["objects"] else None
    valid_areas = [] if "area" in sample["objects"] else None

    if sample['width'] >= 4000 or sample['height'] >= 4000:
        sample["objects"]["bbox"] = valid_bboxes
        sample["objects"]["category"] = valid_categories
        if valid_ids is not None:
            sample["objects"]["id"] = valid_ids
        if valid_areas is not None:
            sample["objects"]["area"] = valid_areas
        
        return sample

    img_width, img_height = sample["image"].size
    category_mappings = {c: i+1 for i, c in enumerate(sorted(tgt_categories))}
    for i, bbox in enumerate(sample["objects"]["bbox"]):
        # x, y, w, h = bbox
        x, y, x2, y2 = bbox
        w = x2 - x + 1
        h = y2 - y + 1
        if all([el >= 0 and el <= img_width for el in [x, x+w]]) and all([el >= 0 and el <= img_height for el in [y, y+h]]) and (len(category_mappings) == 0 or sample["objects"]["category"][i] in category_mappings):
            valid_bboxes.append([x, y, x+w-1, y+h-1])
            valid_categories.append(category_mappings[sample["objects"]["category"][i]])
            if valid_ids is not None:
                valid_ids.append(sample["objects"]["id"][i])
            if valid_areas is not None:
                valid_areas.append(sample["objects"]["area"][i])
                
    sample["objects"]["bbox"] = valid_bboxes
    sample["objects"]["category"] = valid_categories
    if valid_ids is not None:
        sample["objects"]["id"] = valid_ids
    if valid_areas is not None:
        sample["objects"]["area"] = valid_areas
    
    return sample

import torch
from src.utils.anchor_utils import *

import numpy as np
from pycocotools.cocoeval import COCOeval

class CocoEvaluator:
    def __init__(self, coco_gt):
        """
        Args:
            coco_gt (COCO): COCO ground truth object containing the full evaluation annotations.
        """
        self.coco_gt = coco_gt
        self.predictions = []

    def update(self, proposals, bbox_deltas, cls_scores, image_id, im_shape):
        """
        Store predictions for evaluation after decoding bbox deltas.

        Args:
            proposals (Tensor): [N, 5] tensor of proposals (batch_idx, x1, y1, x2, y2).
            bbox_deltas (Tensor): Predicted bbox deltas (shape: [N, 4]).
            cls_scores (Tensor): Predicted classification scores.
            image_id (int): The COCO image ID for this set of predictions.
            im_shape (tuple): (height, width) of the image.
        """
        # Decode final boxes using the proposals (excluding batch index) and bbox deltas.
        decoded_boxes = decode_boxes(proposals[:, 1:], bbox_deltas, im_shape)
        # For each proposal, select the class with the highest probability.
        scores, labels = torch.max(torch.softmax(cls_scores, dim=1), dim=1)
        boxes = decoded_boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        pos_scores = scores[labels == 1]

        # print('='*50)
        # print(pos_scores.min(), pos_scores.max(), pos_scores.mean(), pos_scores.std())
        # quantiles = np.quantile(pos_scores, [0.25, 0.5, 0.75])
        # print("25th percentile:", quantiles[0])
        # print("Median:", quantiles[1])
        # print("75th percentile:", quantiles[2])
        # print(len(labels == 1))
        # print('='*50)

        # Add predictions using the provided image_id
        for box, score, label in zip(boxes, scores, labels):
            if int(label) == 0:
                # Skip predictions where the model predicts background.
                continue

            self.predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1])
                ],
                "score": float(score)
            })

    def compute_AP(self, iou_thresh=None):
        """
        Compute COCO AP metrics. If iou_thresh is provided, override the default IoU thresholds.

        Returns:
            dict: Dictionary containing AP metrics.
                  If iou_thresh is None, returns metrics averaged over the default range.
                  Otherwise returns metrics for that specific IoU threshold.
        """

        coco_dt = self.coco_gt.loadRes(self.predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        if iou_thresh is not None:
            coco_eval.params.iouThrs = np.array([iou_thresh])
        coco_eval.evaluate()
        coco_eval.accumulate()
        # Optionally, you can call coco_eval.summarize() to print the summary.
        coco_eval.summarize()

        if iou_thresh is None:
            # Use default summary indices (using average over multiple thresholds)
            ap_small = coco_eval.stats[3] if len(coco_eval.stats) > 3 else None
            ap_medium = coco_eval.stats[4] if len(coco_eval.stats) > 4 else None
            ap_large = coco_eval.stats[5] if len(coco_eval.stats) > 5 else None
            return {"AP_small": ap_small, "AP_medium": ap_medium, "AP_large": ap_large}
        else:
            # When using a single IoU threshold, stats[0] is the AP at that threshold.
            # The indices for small, medium, large might still be in positions 3, 4, 5.
            ap = coco_eval.stats[0] if len(coco_eval.stats) > 0 else None
            ap_small = coco_eval.stats[3] if len(coco_eval.stats) > 3 else None
            ap_medium = coco_eval.stats[4] if len(coco_eval.stats) > 4 else None
            ap_large = coco_eval.stats[5] if len(coco_eval.stats) > 5 else None
            return {"AP": ap, "AP_small": ap_small, "AP_medium": ap_medium, "AP_large": ap_large}

    def compute_AP_for_thresholds(self, iou_thresholds):
        """
        Compute COCO AP metrics for each IoU threshold provided.

        Args:
            iou_thresholds (list): List of IoU thresholds.

        Returns:
            dict: A dictionary with keys for each IoU threshold containing AP metrics.
                  For example: {"AP_0.50": 0.123, "AP_small_0.50": 0.045, ...}
        """
        results = {}
        for thr in iou_thresholds:
            metrics = self.compute_AP(iou_thresh=thr)
            results[f"AP_{thr:.2f}"] = metrics["AP"]
            results[f"AP_small_{thr:.2f}"] = metrics["AP_small"]
            results[f"AP_medium_{thr:.2f}"] = metrics["AP_medium"]
            results[f"AP_large_{thr:.2f}"] = metrics["AP_large"]
        return results

    def reset(self):
        """Clear predictions."""
        self.predictions.clear()

import torch
import math

import torchvision.ops as ops

def get_scores(rpn_output):
    B, S, H, W = rpn_output.size()
    scores = rpn_output.view(B, -1, 2, H, W)
    scores = scores.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 2)
    return scores.view(-1, 2)

def nms(rois, scores, iou_threshold=0.7, score_threshold=0.05):
    # Apply sigmoid to get probabilities and threshold them.
    keep_initial = scores > score_threshold
    rois = rois[keep_initial]
    scores = scores[keep_initial]
    
    if rois.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=rois.device)
    
    # Use the built-in, optimized NMS.
    keep = ops.nms(rois[:, 1:], scores, iou_threshold)
    return keep

def map_rois_to_fpn_levels(rois, k0=4, canonical_scale=224, min_level=3, max_level=5):
    """
    Map ROIs to FPN levels based on ROI size.
    
    Parameters:
        rois (Tensor): Tensor of shape [N, 5] with each ROI as [batch_idx, x1, y1, x2, y2].
        k0 (int): Base level (typically 4).
        canonical_scale (float): Canonical scale (e.g., 224).
        min_level (int): Minimum FPN level.
        max_level (int): Maximum FPN level.
    
    Returns:
        target_levels (Tensor): Tensor of shape [N] with the FPN level for each ROI.
    """
    # Extract the box coordinates (ignoring batch_idx)
    boxes = rois[:, 1:5]
    
    # Compute width and height. Note the "+1" for inclusive coordinates.
    widths  = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    
    # Compute the scale (sqrt of area)
    scales = torch.sqrt(widths * heights)
    
    # Compute target level using the formula
    target_levels = torch.floor(k0 + torch.log2(scales / canonical_scale + 1e-6))
    
    # Clamp to the valid FPN levels.
    target_levels = torch.clamp(target_levels, min=min_level, max=max_level)
    
    return target_levels.long()

def bilinear_interpolate(feature, x, y):
    """
    Perform bilinear interpolation on a single feature map.

    Parameters:
        feature (Tensor): A tensor of shape [C, H, W].
        x (float): x coordinate (float) in feature map space.
        y (float): y coordinate (float) in feature map space.
    
    Returns:
        Tensor: Interpolated values for each channel [C].
    """
    C, H, W = feature.shape

    # If the point is out-of-bound, return zeros.
    if x < 0 or x > W - 1 or y < 0 or y > H - 1:
        return torch.zeros(C, device=feature.device, dtype=feature.dtype)

    # Find the integer coordinates surrounding (x, y)
    x0 = int(math.floor(x))
    x1 = min(x0 + 1, W - 1)
    y0 = int(math.floor(y))
    y1 = min(y0 + 1, H - 1)

    # Fetch pixel values at these corners
    Ia = feature[:, y0, x0]
    Ib = feature[:, y0, x1]
    Ic = feature[:, y1, x0]
    Id = feature[:, y1, x1]

    # Compute the interpolation weights
    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    # Compute the interpolated feature
    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def perform_roi_align(levels, all_proposals, fpn_features, pooled_height, pooled_width, img_shape):
    assigned_levels = map_rois_to_fpn_levels(all_proposals, min_level=levels[0], max_level=levels[-1])

    N = all_proposals.size(0)

    B, C, H, W = fpn_features['conv' + str(levels[0])].size()

    roi_aligned_features = torch.zeros((N, C, pooled_height, pooled_width), device=all_proposals.device)

    for level in levels:
        level_inds = (assigned_levels == level).nonzero(as_tuple=True)[0]
        if level_inds.numel() == 0:
            continue

        proposals_level = all_proposals[level_inds]
        feature_map = fpn_features['conv' + str(level)]

        spatial_scale = feature_map.size(2) / img_shape[0]

        aligned_features = ops.roi_align(
            feature_map, 
            proposals_level, 
            output_size=(pooled_height, pooled_width), 
            spatial_scale=spatial_scale,
            aligned=True              # Optional: set to True if you want aligned corners.
        )

        roi_aligned_features[level_inds] = aligned_features

    return roi_aligned_features

from tqdm import tqdm
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
import tempfile
import json
import os

from src.utils.anchor_utils import *
from src.utils.roi_utils import *
from src.utils.eval_utils import *

def compute_iou(box, boxes):
    """
    Compute IoU between a single box and a set of boxes.
    
    Args:
        box (Tensor): Tensor of shape [4] for a single box: [x1, y1, x2, y2].
        boxes (Tensor): Tensor of shape [K, 4] for K boxes.
        
    Returns:
        iou (Tensor): Tensor of shape [K] containing IoU values.
    """
    # Intersection coordinates
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    
    inter_w = (x2 - x1 + 1).clamp(min=0)
    inter_h = (y2 - y1 + 1).clamp(min=0)
    inter = inter_w * inter_h

    # Areas of boxes
    area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    
    union = area_box + areas_boxes - inter
    iou = inter / union
    return iou

def bbox_transform(proposals, gt_boxes):
    """
    Compute bounding box regression targets for proposals relative to ground-truth boxes.
    
    Args:
        proposals (Tensor): Tensor of shape [P, 4].
        gt_boxes (Tensor): Tensor of shape [P, 4] (each proposal's matching GT).
        
    Returns:
        targets (Tensor): Tensor of shape [P, 4] containing (dx, dy, dw, dh).
    """
    widths  = proposals[:, 2] - proposals[:, 0] + 1.0
    heights = proposals[:, 3] - proposals[:, 1] + 1.0
    ctr_x   = proposals[:, 0] + 0.5 * widths
    ctr_y   = proposals[:, 1] + 0.5 * heights

    gt_widths  = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x   = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y   = gt_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - ctr_x) / widths
    dy = (gt_ctr_y - ctr_y) / heights
    dw = torch.log(gt_widths / widths)
    dh = torch.log(gt_heights / heights)
    
    targets = torch.stack((dx, dy, dw, dh), dim=1)
    return targets

def compute_iou_vectorized(proposals, gt_boxes):
    """
    Compute the IoU between each proposal and each ground truth box in a vectorized manner.

    Args:
        proposals (Tensor): Tensor of shape [N, 4] for proposals.
        gt_boxes (Tensor): Tensor of shape [M, 4] for ground-truth boxes.

    Returns:
        ious (Tensor): IoU matrix of shape [N, M].
    """
    N = proposals.size(0)
    M = gt_boxes.size(0)

    # proposals: [N,4] and gt_boxes: [M,4]
    # Expand dimensions for broadcasting.
    proposals = proposals.unsqueeze(1)  # shape: [N, 1, 4]
    gt_boxes = gt_boxes.unsqueeze(0)      # shape: [1, M, 4]

    # Compute intersection coordinates.
    x1 = torch.max(proposals[..., 0], gt_boxes[..., 0])  # [N, M]
    y1 = torch.max(proposals[..., 1], gt_boxes[..., 1])
    x2 = torch.min(proposals[..., 2], gt_boxes[..., 2])
    y2 = torch.min(proposals[..., 3], gt_boxes[..., 3])

    inter_w = (x2 - x1 + 1).clamp(min=0)
    inter_h = (y2 - y1 + 1).clamp(min=0)
    inter_area = inter_w * inter_h  # [N, M]

    # Compute areas of proposals and gt_boxes.
    proposal_area = ((proposals[..., 2] - proposals[..., 0] + 1) *
                     (proposals[..., 3] - proposals[..., 1] + 1))  # [N, 1]
    gt_area = ((gt_boxes[..., 2] - gt_boxes[..., 0] + 1) *
               (gt_boxes[..., 3] - gt_boxes[..., 1] + 1))         # [1, M]

    union_area = proposal_area + gt_area - inter_area
    ious = inter_area / union_area
    return ious  # shape: [N, M]

def matching_and_sampling(all_proposals, all_scores, gt, num_samples, pos_iou_thresh=0.5, neg_iou_thresh=0.1, pos_fraction=0.25):
    sampled_proposals_list = []
    batch_scores_list = []
    sampled_labels_list = []
    sampled_bbox_targets_list = []
    sampled_indices_list = []

    batch_ids = all_proposals[:, 0].unique()

    for batch_idx in batch_ids:
        batch_mask = all_proposals[:, 0] == batch_idx

        global_indices = torch.nonzero(batch_mask).squeeze(1)
        batch_proposals = all_proposals[batch_mask][:, 1:]
        batch_scores = all_scores[batch_mask]

        gt_mask = gt[:, 0] == batch_idx
        gt_b = gt[gt_mask][:, 1:5]
        gt_labels = gt[gt_mask][:, 5].long()

        N_b = batch_proposals.shape[0]
        M_b = gt_b.shape[0]

        ious = compute_iou_vectorized(batch_proposals, gt_b)  # [N_b, M_b]
        max_ious, gt_assignment = ious.max(dim=1)  # max_ious: [N_b], gt_assignment: [N_b]

        # Initialize labels for proposals (default background: label 0).
        labels_b = torch.zeros(N_b, dtype=torch.long, device=all_proposals.device)
        # Regression targets are zeros by default.
        bbox_targets_b = torch.zeros((N_b, 4), device=all_proposals.device)

        pos_inds = max_ious >= pos_iou_thresh
        if pos_inds.sum() > 0:
            labels_b[pos_inds] = gt_labels[gt_assignment[pos_inds]]
            pos_proposals = batch_proposals[pos_inds]
            matched_gt = gt_b[gt_assignment[pos_inds]]
            bbox_targets_b[pos_inds] = bbox_transform(pos_proposals, matched_gt)
        
        valid_inds = (max_ious >= pos_iou_thresh) | (max_ious < neg_iou_thresh)

        num_pos = int(pos_fraction * num_samples)
        pos_idx = torch.nonzero(pos_inds).squeeze(1)
        neg_idx = torch.nonzero((~pos_inds) & valid_inds).squeeze(1)

        num_pos_sample = min(num_pos, pos_idx.numel())
        num_neg_sample = num_samples - num_pos_sample

        if pos_idx.numel() > 0:
            perm_pos = pos_idx[torch.randperm(pos_idx.numel())][:num_pos_sample]
        else:
            perm_pos = pos_idx
        if neg_idx.numel() > 0:
            perm_neg = neg_idx[torch.randperm(neg_idx.numel())][:num_neg_sample]
        else:
            perm_neg = neg_idx
        
        keep_inds = torch.cat([perm_pos, perm_neg], dim=0)
        
        sampled_batch_indices = global_indices[keep_inds]
        sampled_indices_list.append(sampled_batch_indices)

        proposals_keep = batch_proposals[keep_inds]  # shape [K_b, 4]
        batch_scores_keep = batch_scores[keep_inds]  # shape [K_b]
        labels_keep = labels_b[keep_inds]         # shape [K_b]
        bbox_targets_keep = bbox_targets_b[keep_inds]  # shape [K_b, 4]
        
        batch_idx_tensor = torch.full((proposals_keep.shape[0], 1), batch_idx.item(), device=all_proposals.device)
        proposals_keep = torch.cat([batch_idx_tensor, proposals_keep], dim=1)  # now [K_b, 5]

        sampled_proposals_list.append(proposals_keep)
        batch_scores_list.append(batch_scores_keep)
        sampled_labels_list.append(labels_keep)
        sampled_bbox_targets_list.append(bbox_targets_keep)

    # Concatenate results from all batches.
    sampled_proposals = torch.cat(sampled_proposals_list, dim=0) if sampled_proposals_list else torch.empty(0, 5, dtype=torch.float32, device=all_proposals.device)
    sampled_scores = torch.cat(batch_scores_list, dim=0) if batch_scores_list else torch.empty(0, 2, dtype=torch.float32, device=all_proposals.device)
    sampled_labels = torch.cat(sampled_labels_list, dim=0) if sampled_labels_list else torch.empty(0, dtype=torch.int64, device=all_proposals.device)
    sampled_bbox_targets = torch.cat(sampled_bbox_targets_list, dim=0) if sampled_bbox_targets_list else torch.empty(0, 4, dtype=torch.float32, device=all_proposals.device)
    sampled_indices = torch.cat(sampled_indices_list, dim=0) if sampled_indices_list else torch.empty(0, dtype=torch.int64, device=all_proposals.device)

    return sampled_proposals, sampled_scores, sampled_labels, sampled_bbox_targets, sampled_indices

def add_batch_idx_to_targets(targets):
    """
    Add batch index as the first column to each target's bounding boxes.

    Args:
        targets (list of dict): Each dict has at least the key "boxes", a tensor of shape [num_boxes, 4].

    Returns:
        list of dict: A new list of target dictionaries with "boxes" of shape [num_boxes, 5],
                      where the first column is the batch index.
    """
    new_targets = [
        {**target, "boxes_with_label": torch.cat([
            torch.full((target["boxes_with_label"].shape[0], 1),
                       i,
                       dtype=target["boxes_with_label"].dtype,
                       device=target["boxes_with_label"].device),
            target["boxes_with_label"]
        ], dim=1)}
        for i, target in enumerate(targets)
    ]
    return new_targets

def rpn_loss_fn(sampled_deltas, sampled_scores, sampled_labels, sampled_bbox_targets):
    cls_loss = F.cross_entropy(sampled_scores, sampled_labels)
    pos_inds = (sampled_labels == 1).nonzero(as_tuple=True)[0]

    if pos_inds.numel() > 0:
        # Use Smooth L1 loss on positive anchors.
        loss_reg = F.smooth_l1_loss(sampled_deltas[pos_inds, 1:],
                                    sampled_bbox_targets[pos_inds],
                                    reduction='sum')
        # Normalize by the number of positive anchors.
        loss_reg = loss_reg / pos_inds.numel()
    else:
        loss_reg = torch.tensor(0.0, device=sampled_scores.device)
    
    return cls_loss, loss_reg

def det_loss_fn(sampled_bbox_preds, sampled_scores, sampled_labels, sampled_bbox_targets, num_classes):
    if sampled_scores.size(0) == 0:
        cls_loss = torch.tensor(0.0, device=sampled_scores.device)
    else:
        cls_loss = F.cross_entropy(sampled_scores, sampled_labels)
    
    pos_inds = torch.nonzero(sampled_labels > 0).squeeze(1)

    if pos_inds.numel() > 0:
        sampled_bbox_preds = sampled_bbox_preds.view(sampled_bbox_preds.size(0), num_classes, 4)
        pos_labels = (sampled_labels[pos_inds] - 1).view(-1, 1, 1).expand(-1, 1, 4)
        sampled_bbox_pred_pos = sampled_bbox_preds[pos_inds].gather(1, pos_labels).squeeze(1)

        # Use Smooth L1 loss on positive anchors.
        loss_reg = F.smooth_l1_loss(sampled_bbox_pred_pos,
                                    sampled_bbox_targets[pos_inds],
                                    reduction='sum')
        # Normalize by the number of positive anchors.
        loss_reg = loss_reg / pos_inds.numel()
    else:
        loss_reg = torch.tensor(0.0, device=sampled_scores.device)
    
    return cls_loss, loss_reg

def train_loop(num_epochs, dataloader, backbone, fpn, rpn, head, optimizer, device,
               layer_to_shifted_anchors, img_shape, num_classes, pooled_height, pooled_width, train_logger, config):
    """
    Unified training loop for the entire detection network using a single optimizer,
    with logging of metrics to a file.
    
    Args:
        num_epochs (int): Number of epochs.
        dataloader (DataLoader): Training data loader.
        backbone (nn.Module): Backbone network.
        fpn (nn.Module): Feature Pyramid Network.
        rpn (nn.Module): Region Proposal Network.
        head (nn.Module): Detection head.
        optimizer (Optimizer): Single optimizer for all parameters.
        device (torch.device): The computation device.
        layer_to_shifted_anchors (dict): Precomputed shifted anchors per FPN level.
        img_shape (tuple): The input image dimensions.
        num_classes (int): Number of object classes.
    """
    backbone.train()
    fpn.train()
    rpn.train()
    head.train()

    for epoch in range(num_epochs):
        for i, (images, targets) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            # Move images and targets to the device
            images = images.to(device)
            targets = add_batch_idx_to_targets(targets)
            # print(images.size(), targets[0]['boxes_with_label'].size(), flush=True)
            gt = torch.cat([t['boxes_with_label'] for t in targets], dim=0).to(device)
            print('='*50)
            print(gt.size())
            print('='*50)
            rpn_gt = gt.clone()
            # Convert labels to binary for the RPN (foreground vs background)
            rpn_gt[:, 5] = 1

            # Forward pass: Backbone -> FPN.
            features = backbone(images)
            fpn_features = fpn(features)

            # Forward pass through the RPN at each FPN level.
            rpn_out = {}
            for k, feature_map in fpn_features.items():
                rpn_out[k] = rpn(feature_map)
            sorted_rpn_keys = sorted(rpn_out.keys())

            # Aggregate RPN outputs.
            all_rpn_deltas = torch.cat(
                [collect_rpn_deltas(rpn_out[k][1]) for k in sorted_rpn_keys], dim=0
            )
            rois = torch.cat(
                [generate_rois(rpn_out[k][1], layer_to_shifted_anchors[k], img_shape)
                 for k in sorted_rpn_keys], dim=0
            )
            scores = torch.cat(
                [get_scores(rpn_out[k][0]) for k in sorted_rpn_keys], dim=0
            )

            # Matching and sampling for RPN training.
            (rpn_sampled_proposals, rpn_sampled_scores, rpn_sampled_labels,
             rpn_sampled_bbox_targets, rpn_sampled_indices) = matching_and_sampling(
                 rois, scores, rpn_gt, config['rpn']['rpn_train_batch_size'], pos_iou_thresh=config['rpn']['rpn_pos_iou_thresh'], neg_iou_thresh=config['rpn']['rpn_neg_iou_thresh'], pos_fraction=config['rpn']['rpn_pos_ratio']
             )
            rpn_sampled_deltas = all_rpn_deltas[rpn_sampled_indices]

            # Process proposals for the detection head.
            unique_batches = rois[:, 0].unique()
            all_proposals = []
            all_scores = []
            for batch_idx in unique_batches:
                batch_mask = rois[:, 0] == batch_idx
                batch_rois = rois[batch_mask]
                batch_scores = scores[batch_mask]
                batch_scores = F.softmax(batch_scores, dim=1)

                # Apply Non-Maximum Suppression (NMS) per batch.
                keep = nms(batch_rois, batch_scores[:, 1], iou_threshold=config['rpn']['nms_iou_thresh'], score_threshold=config['rpn']['nms_score_thresh'])
                batch_rois = batch_rois[keep]
                batch_scores = batch_scores[keep]

                # Sort proposals by score.
                _, sorted_indices = torch.sort(batch_scores[:, 1], descending=True)
                sorted_proposals = batch_rois[sorted_indices]
                sorted_scores = batch_scores[sorted_indices]

                K = config['rpn']['nms_topk_train']  # Adjust K based on your requirements.
                topk_proposals = sorted_proposals[:K]
                topk_scores = sorted_scores[:K]

                all_proposals.append(topk_proposals)
                all_scores.append(topk_scores)

            all_proposals = torch.cat(all_proposals, dim=0).detach()
            all_scores = torch.cat(all_scores, dim=0).detach()

            # Matching and sampling for the detection head.
            (sampled_proposals, sampled_scores, sampled_labels,
             sampled_bbox_targets, _) = matching_and_sampling(
                 all_proposals, all_scores, gt, config['head']['detection_train_batch_size'], pos_iou_thresh=config['head']['detection_pos_iou_thresh'], neg_iou_thresh=config['head']['detection_neg_iou_thresh'], pos_fraction=config['head']['detection_pos_ratio']
             )
            # ROI Align: pool features corresponding to the proposals.
            levels = sorted([int(x[-1]) for x in rpn_out.keys()])
            aligned_proposals = perform_roi_align(
                levels, sampled_proposals, fpn_features,
                pooled_height, pooled_width, img_shape
            )

            # Forward pass through the detection head.
            cls_scores, bbox_deltas = head(aligned_proposals)

            # Compute losses.
            rpn_cls_loss, rpn_bbox_loss = rpn_loss_fn(
                rpn_sampled_deltas, rpn_sampled_scores,
                rpn_sampled_labels, rpn_sampled_bbox_targets
            )
            det_cls_loss, det_bbox_loss = det_loss_fn(
                bbox_deltas, cls_scores,
                sampled_labels, sampled_bbox_targets, num_classes
            )
            total_loss = rpn_cls_loss + rpn_bbox_loss + det_cls_loss + det_bbox_loss

            # Backpropagation and parameter update.
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Build log message
            log_message = (
                f"Epoch: {epoch+1}, Iteration: {i+1}, Total Loss: {total_loss.item():.4f}, "
                f"RPN_cls: {rpn_cls_loss.item():.4f}, RPN_bbox: {rpn_bbox_loss.item():.4f}, "
                f"DET_cls: {det_cls_loss.item():.4f}, DET_bbox: {det_bbox_loss.item():.4f}"
            )
            # Log to file
            train_logger.info(log_message)

            if i % 25 == 0:
                torch.save({
                    'epoch': epoch + 1,
                    'backbone_state_dict': backbone.state_dict(),
                    'fpn_state_dict': fpn.state_dict(),
                    'rpn_state_dict': rpn.state_dict(),
                    'head_state_dict': head.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, f"checkpoints/model_epoch_{epoch+1}_iter_{i+1}.pth")

def validation_loop(dataloader, backbone, fpn, rpn, head, device,
                        layer_to_shifted_anchors, img_shape, num_classes,
                        pooled_height, pooled_width, val_logger, config, ap_iou_thresholds):
    """
    Validation loop for the detection network that computes mAP metrics only (no losses).
    This loop runs inference on the validation set, collects predictions, and evaluates
    detection performance using COCO metrics.

    Args:
        dataloader (DataLoader): Validation data loader.
        backbone (nn.Module): Backbone network.
        fpn (nn.Module): Feature Pyramid Network.
        rpn (nn.Module): Region Proposal Network.
        head (nn.Module): Detection head.
        device (torch.device): The computation device.
        layer_to_shifted_anchors (dict): Precomputed shifted anchors per FPN level.
        img_shape (tuple): The input image dimensions.
        num_classes (int): Number of object classes.
        pooled_height (int): Height for ROI Align.
        pooled_width (int): Width for ROI Align.
        val_logger: Logger for validation metrics.
        config (dict): Configuration dictionary containing RPN and head settings.
        ap_iou_thresholds (list): List of IoU thresholds at which to compute AP metrics.
    
    Returns:
        dict: Dictionary containing AP metrics (mAP values).
    """
    backbone.eval()
    fpn.eval()
    rpn.eval()
    head.eval()

    # Initialize the evaluator (we'll fill in the ground truth later)
    coco_evaluator = CocoEvaluator(None)

    # Prepare COCO-style ground truth annotations.
    annotations = []
    images = []
    categories = [{"id": i, "name": f"class_{i}"} for i in range(1, num_classes + 1)]
    ann_id = 0
    image_id_map = {}  # Maps dataset index to COCO image_id

    # Disable gradient computation.
    with torch.no_grad():
        for i, (image_batch, targets) in enumerate(tqdm(dataloader, desc="Validation")):
            # Move images to device.
            image_batch = image_batch.to(device)
            targets = add_batch_idx_to_targets(targets)

            # Process ground truth to build COCO annotations.
            for batch_idx, target in enumerate(targets):
                img_idx = i * image_batch.shape[0] + batch_idx
                if img_idx not in image_id_map:
                    image_id = len(image_id_map) + 1  # COCO image IDs start from 1.
                    image_id_map[img_idx] = image_id
                    images.append({
                        "id": image_id,
                        "width": img_shape[1],
                        "height": img_shape[0]
                    })
                else:
                    image_id = image_id_map[img_idx]
                boxes = target['boxes_with_label'].cpu()
                for box_idx in range(boxes.shape[0]):
                    x1, y1, x2, y2 = boxes[box_idx, 1:5].tolist()
                    width = x2 - x1
                    height = y2 - y1
                    category_id = int(boxes[box_idx, 5].item())
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, width, height],
                        "area": width * height,
                        "iscrowd": 0
                    })

            # Forward pass: Backbone -> FPN.
            features = backbone(image_batch)
            fpn_features = fpn(features)

            # Run the RPN over each FPN level.
            rpn_out = {}
            for k, feature_map in fpn_features.items():
                rpn_out[k] = rpn(feature_map)
            sorted_rpn_keys = sorted(rpn_out.keys())

            # Aggregate RPN outputs to form proposals.
            rois = torch.cat(
                [generate_rois(rpn_out[k][1], layer_to_shifted_anchors[k], img_shape)
                 for k in sorted_rpn_keys], dim=0
            )
            scores = torch.cat(
                [get_scores(rpn_out[k][0]) for k in sorted_rpn_keys], dim=0
            )

            # For each image in the batch, apply NMS and select top-k proposals.
            unique_batches = rois[:, 0].unique()
            proposals_list = []
            for batch_idx in unique_batches:
                batch_mask = rois[:, 0] == batch_idx
                batch_rois = rois[batch_mask]
                batch_scores = scores[batch_mask]
                batch_scores = F.softmax(batch_scores, dim=1)

                # Apply Non-Maximum Suppression (NMS) per batch.
                keep = nms(batch_rois, batch_scores[:, 1],
                           iou_threshold=config['rpn']['nms_iou_thresh'],
                           score_threshold=config['rpn']['nms_score_thresh'])
                batch_rois = batch_rois[keep]
                batch_scores = batch_scores[keep]

                # Sort proposals by score and keep the top-K.
                _, sorted_indices = torch.sort(batch_scores[:, 1], descending=True)
                sorted_proposals = batch_rois[sorted_indices]
                K = config['rpn']['nms_topk_test']
                topk_proposals = sorted_proposals[:K]
                proposals_list.append(topk_proposals)

            if len(proposals_list) > 0:
                all_proposals = torch.cat(proposals_list, dim=0).detach()
            else:
                continue
            
            # print('='*10)
            # print(len(all_proposals))
            # print('='*10)

            # For each image in the batch, run the detection head to obtain final predictions.
            levels = sorted([int(x[-1]) for x in rpn_out.keys()])
            for batch_idx in unique_batches:
                img_idx = i * image_batch.shape[0] + int(batch_idx.item())
                image_id = image_id_map[img_idx]

                batch_mask = all_proposals[:, 0] == batch_idx
                batch_proposals = all_proposals[batch_mask]
                if batch_proposals.shape[0] == 0:
                    continue

                # ROI Align to extract fixed-size features for each proposal.
                batch_aligned_features = perform_roi_align(
                    levels, batch_proposals, fpn_features,
                    pooled_height, pooled_width, img_shape
                )
                # Run detection head on these aligned features.
                batch_cls_scores, batch_bbox_deltas = head(batch_aligned_features)

                # Update the COCO evaluator with the predictions - FIX: Pass image_id directly
                coco_evaluator.update(
                    batch_proposals,
                    batch_bbox_deltas,
                    batch_cls_scores,
                    image_id,
                    img_shape
                )

    # After looping through the dataloader, create the COCO ground truth.
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f)
        temp_file_name = f.name

    coco_gt = COCO(temp_file_name)
    coco_evaluator.coco_gt = coco_gt

    # Compute AP metrics for each specified IoU threshold.
    ap_metrics = coco_evaluator.compute_AP_for_thresholds(ap_iou_thresholds)
    ap_metrics_str = "\n".join(
        [f"{key:<20}: {value:.4f}" for key, value in sorted(ap_metrics.items())]
    )
    val_logger.info("COCO Evaluation AP Metrics:\n" + ap_metrics_str)

    # Clean up the temporary file.
    try:
        os.unlink(temp_file_name)
    except Exception:
        pass

    return ap_metrics