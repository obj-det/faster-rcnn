"""
Configuration utilities for Faster R-CNN training and inference.

This module provides utility functions for setting up all components of the Faster R-CNN
training pipeline based on a YAML configuration file. It handles:
- Dataset loading and preprocessing
- Model initialization
- Optimizer setup
- Data augmentation
- Logging configuration
- Anchor generation
- Checkpoint management
"""

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

from src.backbones import Backbone, ResNetBackbone
from src.fpn import FPN
from src.rpn import RPN
from src.head import DetectionHead

from src.utils.anchor_utils import generate_anchors, shift

import datasets
from src.utils.data_utils import filter_bboxes_in_sample

def load_config(config_path):
    """Load configuration from YAML file.
    
    Args:
        config_path (str): Path to the YAML configuration file.
    
    Returns:
        dict: Configuration dictionary containing all parameters.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def setup_dataset(config):
    """Set up and return datasets based on configuration.
    
    This function:
    1. Loads the dataset specified in the config
    2. Filters objects based on target categories
    3. Balances positive and negative samples
    4. Saves sample images for visualization
    
    Args:
        config (dict): Configuration dictionary containing dataset parameters.
    
    Returns:
        tuple: (filtered_train_ds, filtered_val_ds) containing the processed datasets.
    """
    ds = datasets.load_dataset(config['dataset']['name'])
    train_ds = ds['train']
    val_ds = ds['val']

    tgt_categories = config['dataset']['tgt_categories']
    load_from_cache_file = config['dataset']['load_from_cache_file']
    
    # Filter bboxes based on target categories
    mapped_train_ds = train_ds.map(lambda sample: filter_bboxes_in_sample(sample, tgt_categories), load_from_cache_file=load_from_cache_file)
    mapped_val_ds = val_ds.map(lambda sample: filter_bboxes_in_sample(sample, tgt_categories), load_from_cache_file=load_from_cache_file)
    
    # Split into positive and negative samples
    train_positive_ds = mapped_train_ds.filter(lambda sample: len(sample["objects"]["bbox"]) > 0, load_from_cache_file=load_from_cache_file)
    train_negative_ds = mapped_train_ds.filter(lambda sample: len(sample["objects"]["bbox"]) == 0, load_from_cache_file=load_from_cache_file)
    
    # Balance negative samples (1:4 ratio)
    neg_samples = min(len(train_positive_ds) // 4, len(train_negative_ds))
    train_negative_ds_balanced = train_negative_ds.shuffle(seed=42).select(range(neg_samples))

    filtered_train_ds = datasets.concatenate_datasets([train_positive_ds, train_negative_ds_balanced]).shuffle(seed=42)

    # Repeat for validation set
    val_positive_ds = mapped_val_ds.filter(lambda sample: len(sample["objects"]["bbox"]) > 0, load_from_cache_file=load_from_cache_file)
    val_negative_ds = mapped_val_ds.filter(lambda sample: len(sample["objects"]["bbox"]) == 0, load_from_cache_file=load_from_cache_file)

    neg_samples = min(len(val_positive_ds) // 4, len(val_negative_ds))
    val_negative_ds_balanced = val_negative_ds.shuffle(seed=42).select(range(neg_samples))

    filtered_val_ds = datasets.concatenate_datasets([val_positive_ds, val_negative_ds_balanced]).shuffle(seed=42)

    # Save sample images
    filtered_train_ds[0]['image'].save('sample_train.jpg')
    filtered_val_ds[0]['image'].save('sample_val.jpg')

    # Count categories for statistics
    category_to_count = defaultdict(int)
    for sample in filtered_train_ds:
        for obj in sample["objects"]["category"]:
            category_to_count[obj] += 1
    
    print(category_to_count)
    print(len(filtered_train_ds), len(filtered_val_ds))

    return filtered_train_ds, filtered_val_ds

def setup_preprocess_transform(config):
    """Create a preprocessing transform pipeline.
    
    Creates a torchvision transform that:
    1. Converts PIL images to tensors
    2. Normalizes the tensors using mean and std from config
    
    Args:
        config (dict): Configuration dictionary containing image normalization parameters.
    
    Returns:
        transforms.Compose: Preprocessing transform pipeline.
    """
    preprocess_pipeline = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=config['image']['mean'],
            std=config['image']['std']
        )
    ])
    return preprocess_pipeline

def setup_augmentation_transform(config, mode):
    """Create an augmentation pipeline with Albumentations.
    
    Sets up image augmentations based on the configuration and mode (train/test).
    Supports various augmentations including:
    - Horizontal/Vertical flips
    - Random crops
    - Brightness/Contrast adjustments
    - Hue/Saturation adjustments
    
    Args:
        config (dict): Configuration dictionary containing augmentation parameters.
        mode (str): Either 'train' or 'test', determines which augmentations to apply.
    
    Returns:
        A.Compose: Albumentations transform pipeline with bbox support.
    """
    aug_transforms = []

    if mode == 'train':
        # Add training-specific augmentations
        if config['augmentation'].get('horizontal_flip_prob', 0) > 0:
            aug_transforms.append(A.HorizontalFlip(p=config['augmentation']['horizontal_flip_prob']))

        if config['augmentation'].get('vertical_flip_prob', 0) > 0:
            aug_transforms.append(A.VerticalFlip(p=config['augmentation']['vertical_flip_prob']))

        # Add color augmentations
        if config['augmentation'].get('brightness_range', []) or config['augmentation'].get('contrast_range', []):
            brightness_limit = (config['augmentation']['brightness_range'][0] - 1, config['augmentation']['brightness_range'][1] - 1)
            contrast_limit = (config['augmentation']['contrast_range'][0] - 1, config['augmentation']['contrast_range'][1] - 1)
            aug_transforms.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.5))

        if config['augmentation'].get('hue_range', []) or config['augmentation'].get('saturation_range', []):
            aug_transforms.append(A.HueSaturationValue(
                hue_shift_limit=int(config['augmentation']['hue_range'][1]*100),
                sat_shift_limit=int((config['augmentation']['saturation_range'][1]-1)*100),
                val_shift_limit=0,
                p=0.5
            ))
    
    # Always resize to target shape
    height, width = config['image']['shape']
    aug_transforms.append(A.Resize(height, width))
    
    transform_pipeline = A.Compose(
        aug_transforms,
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["category"])
    )
    return transform_pipeline

def setup_models(config, num_classes, device, ckpt_path=None):
    """Initialize and configure all model components.
    
    Sets up the complete Faster R-CNN model architecture:
    1. Backbone network (VGG16 or ResNet101)
    2. Feature Pyramid Network (FPN)
    3. Region Proposal Network (RPN)
    4. Detection Head
    
    Args:
        config (dict): Configuration dictionary containing model parameters.
        num_classes (int): Number of object classes to detect.
        device (torch.device): Device to place the models on.
        ckpt_path (str, optional): Path to checkpoint file for loading weights.
    
    Returns:
        tuple: (backbone, fpn, rpn, head) containing all model components.
    """
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
    if backbone_config['type'] == 'vgg16':
        backbone = Backbone(output_layer_map).to(device)
    elif backbone_config['type'] == 'resnet101':
        backbone = ResNetBackbone(output_layer_map).to(device)
    else:
        raise ValueError(f"Unsupported backbone: {config['backbone']['type']}")
    
    fpn = FPN(in_channels=fpn_in_channels, out_channels=fpn_out_channels).to(device)
    
    # Setup RPN
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
    
    # Load checkpoint if provided
    if ckpt_path:
        ckpt = torch.load(ckpt_path, weights_only=False)
        backbone.load_state_dict(ckpt['backbone_state_dict'])
        fpn.load_state_dict(ckpt['fpn_state_dict'])
        rpn.load_state_dict(ckpt['rpn_state_dict'])
        head.load_state_dict(ckpt['head_state_dict'])

    return backbone, fpn, rpn, head

def setup_optimizer(config, models):
    """Set up optimizer for all model components.
    
    Supports Adam and SGD optimizers with configurable parameters.
    
    Args:
        config (dict): Configuration dictionary containing optimizer parameters.
        models (tuple): (backbone, fpn, rpn, head) model components.
    
    Returns:
        torch.optim.Optimizer: Configured optimizer.
    """
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
    """Set up training and validation loggers.
    
    Creates two separate loggers for training and validation metrics,
    with configurable log directories and file names.
    
    Args:
        config (dict): Configuration dictionary containing logging parameters.
    
    Returns:
        tuple: (train_logger, val_logger) configured logging objects.
    """
    log_dir = config['logging']['log_dir']
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Get log filenames
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
    val_logger.propagate = False
    
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