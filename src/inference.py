import os
import torch
import argparse
from src.utils.config_utils import (
    load_config, 
    setup_preprocess_transform,
    setup_augmentation_transform,
    setup_models, 
    setup_anchors
)
from src.utils.inference_utils import *

def parse_args():
    parser = argparse.ArgumentParser(description='Train object detection model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to configuration file')
    parser.add_argument('--checkpoint', type=str, default=None, help='Path to model checkpoint')
    parser.add_argument('--input-dir', type=str, default='data/images', help='Directory with input images')
    parser.add_argument('--output-dir', type=str, default='output', help='Directory where output images will be saved')
    return parser.parse_args()

def main():
    # Parse arguments and load configuration
    args = parse_args()
    config = load_config(args.config)
    checkpoint = args.checkpoint
    input_dir = args.input_dir
    output_dir = args.output_dir
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup model components
    num_classes = len(config['dataset']['tgt_categories'])
    models = setup_models(config, num_classes, device, ckpt_path=checkpoint)
    backbone, fpn, rpn, head = models

    # Setup anchors
    layer_to_shifted_anchors = setup_anchors(config, device)

    # Setup transforms and preprocessing
    preprocess_transform = setup_preprocess_transform(config)
    augmentation_transform_test = setup_augmentation_transform(config, mode='test')

    head_config = config['model']['head']
    pooled_height = head_config['pooled_height']
    pooled_width = head_config['pooled_width']
    category_names = config['dataset']['category_names']

    # Run inference on the input directory
    inference_on_directory(input_dir, output_dir, category_names, backbone, fpn, rpn, head,
                           layer_to_shifted_anchors, augmentation_transform_test, preprocess_transform,
                           device, pooled_height,
                           pooled_width, config['model'])

if __name__ == "__main__":
    main()

