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
    
    # Count categories for determining num_classes
    category_to_count = defaultdict(int)
    for sample in filtered_train_ds:
        for obj in sample["objects"]["category"]:
            category_to_count[obj] += 1
    
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
    
    # Resize using the image shape defined in config.
    width, height = config['image']['shape']
    aug_transforms.append(A.Resize(height, width))
    
    if mode == 'train':
        # Apply horizontal flip if specified.
        if config['augmentation'].get('horizontal_flip_prob', 0) > 0:
            aug_transforms.append(A.HorizontalFlip(p=config['augmentation']['horizontal_flip_prob']))
        
        # Apply vertical flip if specified.
        if config['augmentation'].get('vertical_flip_prob', 0) > 0:
            aug_transforms.append(A.VerticalFlip(p=config['augmentation']['vertical_flip_prob']))
        
        # Random crop augmentation (you may adjust crop size as needed).
        if config['augmentation'].get('random_crop_prob', 0) > 0:
            # Example: crop to 80% of the original size.
            crop_width = int(0.8 * width)
            crop_height = int(0.8 * height)
            aug_transforms.append(A.RandomCrop(width=crop_width, height=crop_height, p=config['augmentation']['random_crop_prob']))
        
        # Brightness and contrast adjustment.
        if config['augmentation'].get('brightness_range', []) or config['augmentation'].get('contrast_range', []):
            # Calculate limits: Albumentations expects limits relative to zero.
            brightness_limit = (config['augmentation']['brightness_range'][0] - 1, config['augmentation']['brightness_range'][1] - 1)
            contrast_limit = (config['augmentation']['contrast_range'][0] - 1, config['augmentation']['contrast_range'][1] - 1)
            aug_transforms.append(A.RandomBrightnessContrast(brightness_limit=brightness_limit, contrast_limit=contrast_limit, p=0.5))
        
        # Saturation and hue adjustment.
        if config['augmentation'].get('hue_range', []) or config['augmentation'].get('saturation_range', []):
            # Note: Hue and saturation limits might need scaling; adjust as appropriate.
            aug_transforms.append(A.HueSaturationValue(
                hue_shift_limit=int(config['augmentation']['hue_range'][1]*100),  
                sat_shift_limit=int((config['augmentation']['saturation_range'][1]-1)*100),
                val_shift_limit=0,
                p=0.5
            ))
    
    # Create the Albumentations pipeline with bbox parameters.
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
        ckpt = torch.load(ckpt_path)
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