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
    """
    Runs inference on all images in input_dir and saves output images with
    bounding boxes drawn for each detection into output_dir.
    
    This version selects the top_k detections per category based on the final
    classification scores.
    
    Args:
        input_dir (str): Directory with input images.
        output_dir (str): Directory where output images will be saved.
        backbone, fpn, rpn, head: Model components.
        layer_to_shifted_anchors (dict): Precomputed shifted anchors per FPN level.
        preprocess_transform: Torchvision transform for preprocessing.
        device (torch.device): Computation device.
        pooled_height, pooled_width (int): ROI Align output dimensions.
        config (dict): Configuration parameters (used for NMS thresholds, etc.).
        score_threshold (float): Minimum class score to keep a detection.
        topk (int): Number of top proposals to keep per category.
    """
    backbone.eval()
    fpn.eval()
    rpn.eval()
    head.eval()

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # List image files (assuming common image extensions)
    image_files = [os.path.join(input_dir, f)
                   for f in os.listdir(input_dir)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_path in image_files:
        # Load and preprocess the image.
        pil_img = Image.open(image_path).convert("RGB")
        transformed_img = augmentation_transform_test(image=np.array(pil_img), bboxes=[], category=[])
        pil_img = Image.fromarray(transformed_img["image"]).convert("RGB")
        img_tensor = preprocess_transform(pil_img)
        img_tensor = img_tensor.unsqueeze(0).to(device)

        with torch.no_grad():
            # Backbone & FPN
            features = backbone(img_tensor)
            fpn_features = fpn(features)

            # RPN forward pass on each FPN level.
            rpn_out = {}
            for k, feature_map in fpn_features.items():
                rpn_out[k] = rpn(feature_map)
            sorted_rpn_keys = sorted(rpn_out.keys())

            # Aggregate RPN outputs.
            all_rpn_deltas = torch.cat(
                [collect_rpn_deltas(rpn_out[k][1]) for k in sorted_rpn_keys],
                dim=0
            )
            # Use the image shape from the tensor (assumes [B, C, H, W])
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

            # For each image in the batch, apply NMS and select top-k proposals.
            batch_rois = rois
            batch_scores = scores
            batch_scores = F.softmax(batch_scores, dim=1)

            _, pre_nms_sorted_indices = torch.sort(batch_scores[:, 1], descending=True)
            batch_rois = batch_rois[pre_nms_sorted_indices]
            batch_scores = batch_scores[pre_nms_sorted_indices]

            batch_rois = batch_rois[:config['rpn']['pre_nms_topk_test']]
            batch_scores = batch_scores[:config['rpn']['pre_nms_topk_test']]

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
            all_proposals = sorted_proposals[:K]

            levels = sorted([int(x[-1]) for x in rpn_out.keys()])
            
            # Filter proposals for this image
            batch_proposals = all_proposals
            if batch_proposals.shape[0] == 0:
                continue

            # ROI Align
            aligned_feats = perform_roi_align(
                levels, batch_proposals, fpn_features,
                pooled_height, pooled_width, img_shape
            )
            # Detection head
            batch_cls_scores, batch_bbox_deltas = head(aligned_feats)

            decoded_boxes = decode_boxes(batch_proposals[:, 1:], batch_bbox_deltas, img_shape)

            all_probs = F.softmax(batch_cls_scores, dim=1)
            confs, labels = torch.max(all_probs, dim=1)

            fg_mask = (labels > 0) & (confs >= config['head']['detection_test_score_thresh'])
            final_boxes = decoded_boxes[fg_mask]
            final_scores = confs[fg_mask]
            final_labels = labels[fg_mask]

            if final_boxes.numel() == 0:
                continue

            final_keep = ops.nms(final_boxes, final_scores, config['head']['detection_test_nms_iou_thresh'])
            final_boxes = final_boxes[final_keep].cpu()
            final_scores = final_scores[final_keep].cpu()
            final_labels = final_labels[final_keep].cpu()

        # Convert original image to NumPy array for plotting.
        img_np = np.array(pil_img)

        # Create a matplotlib figure.
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img_np)

        # Generate a dynamic color map based on unique labels.
        unique_labels = np.unique(final_labels)
        n_labels = len(unique_labels)
        cmap = plt.cm.get_cmap('Set1', n_labels)  # Using Set1 instead of tab20 for better contrast.
        color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
        default_color = (0, 1, 0, 1)  # Bright green as a fallback if a label is not found.

        # Draw each bounding box.
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

        # Save the image with detections.
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Inference complete. Processed images saved to {output_dir}.")