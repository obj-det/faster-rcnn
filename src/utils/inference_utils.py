"""
Inference utilities for Faster R-CNN object detection.

This module provides utilities for running inference with a trained Faster R-CNN model:
- Batch inference on directories of images
- Visualization of detection results
- Post-processing of model outputs including NMS and score thresholding
- Support for multiple object categories with color-coded visualization
"""

import os
import json
import tempfile
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from tqdm import tqdm

from . import *

# -----------------------------------------------------------------------------
# Inference on a Directory of Images
# -----------------------------------------------------------------------------
def inference_on_directory(input_dir, output_dir, category_names, backbone, fpn, rpn, head,
                           layer_to_shifted_anchors, augmentation_transform_test, preprocess_transform,
                           device, pooled_height, pooled_width, config):
    """Run inference on all images in a directory and save visualized results.
    
    This function:
    1. Processes all images in the input directory
    2. Runs the complete Faster R-CNN inference pipeline:
       - Feature extraction with backbone and FPN
       - Region proposal generation with RPN
       - ROI classification and refinement with detection head
    3. Applies post-processing:
       - Non-maximum suppression (NMS)
       - Score thresholding
       - Top-K proposal selection
    4. Visualizes results with color-coded bounding boxes and confidence scores
    
    Args:
        input_dir (str): Directory containing input images (.jpg, .jpeg, .png)
        output_dir (str): Directory to save visualization results
        category_names (list): List of category names for visualization
        backbone (nn.Module): Backbone network for feature extraction
        fpn (nn.Module): Feature Pyramid Network
        rpn (nn.Module): Region Proposal Network
        head (nn.Module): Detection head for classification and box regression
        layer_to_shifted_anchors (dict): Pre-computed anchors for each FPN level
        augmentation_transform_test (A.Compose): Albumentations transform for test-time augmentation
        preprocess_transform (transforms.Compose): Preprocessing transform for backbone
        device (torch.device): Device to run inference on
        pooled_height (int): Height of ROI features after ROI Align
        pooled_width (int): Width of ROI features after ROI Align
        config (dict): Configuration dictionary containing:
            - RPN parameters (pre_nms_topk_test, nms_iou_thresh, etc.)
            - Detection head parameters (detection_test_score_thresh, etc.)
    """
    # Set all models to evaluation mode
    backbone.eval()
    fpn.eval()
    rpn.eval()
    head.eval()

    # Create output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get list of image files
    image_files = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_path in image_files:
        # Load and preprocess image
        pil_img = Image.open(image_path).convert("RGB")
        transformed_img = augmentation_transform_test(image=np.array(pil_img), bboxes=[], category=[])
        pil_img = Image.fromarray(transformed_img["image"]).convert("RGB")
        img_tensor = preprocess_transform(pil_img)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            # Extract features with backbone and FPN
            features = backbone(img_tensor)
            fpn_features = fpn(features)

            # Generate region proposals with RPN
            rpn_out = {}
            for k, feature_map in fpn_features.items():
                rpn_out[k] = rpn(feature_map)
            sorted_rpn_keys = sorted(rpn_out.keys())

            # Aggregate RPN outputs across FPN levels
            all_rpn_deltas = torch.cat(
                [collect_rpn_deltas(rpn_out[k][1]) for k in sorted_rpn_keys],
                dim=0
            )
            img_shape = img_tensor.shape[2:]
            rois = torch.cat(
                [generate_rois(rpn_out[k][1], layer_to_shifted_anchors[k], img_shape)
                 for k in sorted_rpn_keys],
                dim=0
            )
            scores = torch.cat(
                [get_scores(rpn_out[k][0]) for k in sorted_rpn_keys],
                dim=0
            )

            # Process proposals with NMS and top-K selection
            batch_rois = rois
            batch_scores = F.softmax(scores, dim=1)

            # Sort proposals by objectness score
            _, pre_nms_sorted_indices = torch.sort(batch_scores[:, 1], descending=True)
            batch_rois = batch_rois[pre_nms_sorted_indices]
            batch_scores = batch_scores[pre_nms_sorted_indices]

            # Apply pre-NMS top-K filtering
            batch_rois = batch_rois[:config['rpn']['pre_nms_topk_test']]
            batch_scores = batch_scores[:config['rpn']['pre_nms_topk_test']]

            # Apply NMS
            keep = nms(batch_rois, batch_scores[:, 1],
                      iou_threshold=config['rpn']['nms_iou_thresh'],
                      score_threshold=config['rpn']['nms_score_thresh'])
            batch_rois = batch_rois[keep]
            batch_scores = batch_scores[keep]

            # Keep top-K proposals after NMS
            _, sorted_indices = torch.sort(batch_scores[:, 1], descending=True)
            sorted_proposals = batch_rois[sorted_indices]
            K = config['rpn']['nms_topk_test']
            all_proposals = sorted_proposals[:K]

            # Get FPN levels
            levels = sorted([int(x[-1]) for x in rpn_out.keys()])
            
            # Process proposals for current image
            batch_proposals = all_proposals
            if batch_proposals.shape[0] == 0:
                continue

            # Apply ROI Align and detection head
            aligned_feats = perform_roi_align(
                levels, batch_proposals, fpn_features,
                pooled_height, pooled_width, img_shape
            )
            batch_cls_scores, batch_bbox_deltas = head(aligned_feats)

            # Post-process detections
            decoded_boxes = decode_boxes(batch_proposals[:, 1:], batch_bbox_deltas, img_shape)
            all_probs = F.softmax(batch_cls_scores, dim=1)
            confs, labels = torch.max(all_probs, dim=1)

            # Filter by confidence and apply NMS
            fg_mask = (labels > 0) & (confs >= config['head']['detection_test_score_thresh'])
            final_boxes = decoded_boxes[fg_mask]
            final_scores = confs[fg_mask]
            final_labels = labels[fg_mask]

            if final_boxes.numel() == 0:
                continue
            
            final_keep = ops.batched_nms(final_boxes, final_scores, final_labels, iou_threshold=config['head']['detection_test_nms_iou_thresh'])
            final_boxes = final_boxes[final_keep].cpu()
            final_scores = final_scores[final_keep].cpu()
            final_labels = final_labels[final_keep].cpu()

        # Visualize results
        img_np = np.array(pil_img)
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img_np)

        # Create color map for visualization
        unique_labels = np.unique(final_labels)
        n_labels = len(unique_labels)
        cmap = plt.cm.get_cmap('Set1', n_labels)
        color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
        default_color = (0, 1, 0, 1)

        # Draw detections
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            color = color_map.get(label, default_color)
            rect = patches.Rectangle((x1, y1), width, height,
                                  linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"{category_names[label-1]}: {score:.2f}", color=color,
                   fontsize=8, backgroundcolor='white')

        # Save visualization
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Inference complete. Processed images saved to {output_dir}.")