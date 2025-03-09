"""
Training script for Faster R-CNN object detection model.

This script handles the complete training pipeline for the Faster R-CNN model, including:
- Data loading and preprocessing
- Model setup and initialization
- Training and validation loops
- Checkpoint saving and logging
- Augmentation visualization

The script uses a configuration file to specify all training parameters and model architecture.
"""

import os
import torch
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import numpy as np
import argparse
from src.utils.config_utils import (
    load_config, 
    setup_dataset, 
    setup_preprocess_transform,
    setup_augmentation_transform,
    setup_models, 
    setup_optimizer, 
    setup_logging, 
    setup_dataloaders, 
    setup_anchors, 
    setup_checkpoint_dir
)
from src.utils.data_utils import DetectionDataset, collate_fn
from src.utils.training_utils import train_loop, validation_loop

def parse_args():
    """Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed command line arguments containing:
            - config (str): Path to the configuration file
    """
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    return parser.parse_args()

def main():
    """Main training function.
    
    This function orchestrates the entire training process:
    1. Sets up all necessary components (models, optimizers, datasets, etc.)
    2. Runs the training loop for the specified number of epochs
    3. Performs validation after each epoch
    4. Saves checkpoints based on specified criteria
    5. Generates sample visualizations of augmented images
    
    The function uses configuration parameters from a YAML file to control all aspects
    of training, including model architecture, training hyperparameters, and logging settings.
    """
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Setup datasets
    filtered_train_ds, filtered_val_ds = setup_dataset(config)
    num_classes = len(config['dataset']['tgt_categories'])
    print(f"Number of classes: {num_classes}")
    
    # Setup transforms and preprocessing
    preprocess_transform = setup_preprocess_transform(config)
    augmentation_transform_train = setup_augmentation_transform(config, mode='train')
    augmentation_transform_test = setup_augmentation_transform(config, mode='test')
    
    # Setup model components
    models = setup_models(config, num_classes, device)
    backbone, fpn, rpn, head = models
    
    # Setup optimizer
    optimizer = setup_optimizer(config, models)
    
    # Setup logging
    train_logger, val_logger = setup_logging(config)
    
    # Setup datasets and dataloaders
    train_dataset = DetectionDataset(filtered_train_ds, augmentation_transform_train, preprocess_transform)
    val_dataset = DetectionDataset(filtered_val_ds, augmentation_transform_test, preprocess_transform)
    
    # Save sample augmented images for visualization
    save_dir = os.path.join('./', 'sample_images')
    os.makedirs(save_dir, exist_ok=True)

    # Process and save 5 augmented train images
    for i in range(5):
        sample = filtered_train_ds[i]
        image = np.array(sample["image"].convert("RGB"))
        transformed = augmentation_transform_train(image=image, bboxes=[], category=[])
        aug_image = transformed["image"]
        image_pil = Image.fromarray(aug_image)
        image_path = os.path.join(save_dir, f"train_image_{i}.png")
        image_pil.save(image_path)
        print(f"Saved {image_path}")

    # Process and save 5 augmented validation images
    for i in range(5):
        sample = filtered_val_ds[i]
        image = np.array(sample["image"].convert("RGB"))
        transformed = augmentation_transform_test(image=image, bboxes=[], category=[])
        aug_image = transformed["image"]
        image_pil = Image.fromarray(aug_image)
        image_path = os.path.join(save_dir, f"val_image_{i}.png")
        image_pil.save(image_path)
        print(f"Saved {image_path}")

    train_dataloader, val_dataloader = setup_dataloaders(config, train_dataset, val_dataset, collate_fn)
    
    # Setup anchors
    layer_to_shifted_anchors = setup_anchors(config, device)
    
    # Setup checkpoint directory
    checkpoint_dir = setup_checkpoint_dir(config)
    
    # Extract training parameters
    img_shape = tuple(config['image']['shape'])
    head_config = config['model']['head']
    pooled_height = head_config['pooled_height']
    pooled_width = head_config['pooled_width']
    num_epochs = config['training']['num_epochs']
    
    # Training loop
    best_val_loss = float('inf')
    best_ap = 0.0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Training phase
        train_loop(
            epoch+1, config['training']['grad_accumulation_steps'], train_dataloader, backbone, fpn, rpn, head, optimizer, device,
            layer_to_shifted_anchors, img_shape, num_classes, 
            pooled_height, pooled_width, train_logger, config['model']
        )
        
        # Validation phase
        metrics = validation_loop(
            epoch+1, val_dataloader, backbone, fpn, rpn, head, device,
            layer_to_shifted_anchors, img_shape, num_classes, 
            pooled_height, pooled_width, val_logger, config['model'], config['evaluation']['ap_iou_thresholds']
        )
        
        # Get validation metrics
        val_loss = metrics['avg_loss'] if 'avg_loss' in metrics else float('inf')
        ap_metrics = {k: v for k, v in metrics.items() if 'AP' in k}
        
        # Handle checkpoint saving
        save_frequency = config['checkpoints']['save_frequency']
        save_best = config['checkpoints']['save_best']
        should_save_frequency = (epoch + 1) % save_frequency == 0
        
        # Check if current model is the best so far
        metric_to_monitor = config['checkpoints']['metric_to_monitor']
        is_better = False
        
        if metric_to_monitor == 'val_loss' and val_loss < best_val_loss:
            best_val_loss = val_loss
            is_better = True
        elif 'AP' in metric_to_monitor and metric_to_monitor in ap_metrics:
            if ap_metrics[metric_to_monitor] > best_ap:
                best_ap = ap_metrics[metric_to_monitor]
                is_better = True
        
        should_save_best = save_best and is_better
        
        # Save periodic checkpoint
        if should_save_frequency:
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1,
                'backbone_state_dict': backbone.state_dict(),
                'fpn_state_dict': fpn.state_dict(),
                'rpn_state_dict': rpn.state_dict(),
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'ap_metrics': ap_metrics if ap_metrics else None,
                'config': config,
            }, checkpoint_path)
            print(f"Saved checkpoint at {checkpoint_path}")

        # Save best model checkpoint
        if should_save_best:
            checkpoint_path = os.path.join(checkpoint_dir, "model_best.pth")
            torch.save({
                'epoch': epoch + 1,
                'backbone_state_dict': backbone.state_dict(),
                'fpn_state_dict': fpn.state_dict(),
                'rpn_state_dict': rpn.state_dict(),
                'head_state_dict': head.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'ap_metrics': ap_metrics if ap_metrics else None,
                'config': config,
            }, checkpoint_path)
            print(f"Saved best checkpoint at {checkpoint_path}")

if __name__ == "__main__":
    main()