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

from src.utils import *

# -----------------------------------------------------------------------------
# Inference on a Directory of Images
# -----------------------------------------------------------------------------
def inference_on_directory(input_dir, output_dir, backbone, fpn, rpn, head,
                           layer_to_shifted_anchors, preprocess_transform,
                           device, pooled_height, pooled_width, config,
                           score_threshold=0.1, topk=10):
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

            # Process proposals: since we have one image, unique_batches should be [0]
            unique_batches = rois[:, 0].unique()
            all_proposals = []
            all_scores = []
            for batch_idx in unique_batches:
                batch_mask = rois[:, 0] == batch_idx
                batch_rois = rois[batch_mask]
                batch_scores = scores[batch_mask]

                # Apply NMS (using thresholds from config; adjust keys if needed)
                keep = nms(
                    batch_rois,
                    batch_scores[:, 1],
                    iou_threshold=config['rpn']['nms_iou_thresh'],
                    score_threshold=config['rpn']['nms_score_thresh']
                )
                batch_rois = batch_rois[keep]
                batch_scores = batch_scores[keep]

                # Sort proposals by score (descending)
                _, sorted_indices = torch.sort(batch_scores[:, 1], descending=True)
                sorted_proposals = batch_rois[sorted_indices]
                sorted_scores = batch_scores[sorted_indices]

                topk_proposals = sorted_proposals[:config['rpn']['nms_topk_test']]  # initial limit per image if desired
                topk_scores = sorted_scores[:config['rpn']['nms_topk_test']]

                all_proposals.append(topk_proposals)
                all_scores.append(topk_scores)

            all_proposals = torch.cat(all_proposals, dim=0)
            all_scores = torch.cat(all_scores, dim=0)

            # ROI Align: pool features corresponding to the proposals.
            levels = sorted([int(x[-1]) for x in rpn_out.keys()])
            aligned_proposals = perform_roi_align(
                levels, all_proposals, fpn_features,
                pooled_height, pooled_width, img_shape
            )

            # Detection head
            cls_scores, bbox_deltas = head(aligned_proposals)
            decoded_boxes = decode_boxes(all_proposals[:, 1:], bbox_deltas, img_shape)
            cls_scores = torch.softmax(cls_scores, dim=1)

            # Compute final predicted labels and scores for each detection.
            final_labels = torch.argmax(cls_scores, dim=1)
            final_scores = cls_scores[torch.arange(cls_scores.size(0)), final_labels]
            
            # Filter detections below the score threshold.
            keep = final_scores > score_threshold
            final_boxes = decoded_boxes[keep]
            final_scores = final_scores[keep]
            final_labels = final_labels[keep]

            # Now, select top_k detections per category based on the final scores.
            selected_indices = []
            unique_labels = torch.unique(final_labels)
            for label in unique_labels:
                cat_indices = (final_labels == label).nonzero(as_tuple=True)[0]
                if cat_indices.numel() == 0:
                    continue
                cat_scores = final_scores[cat_indices]
                _, order = torch.sort(cat_scores, descending=True)
                selected_cat_indices = cat_indices[order][:topk]
                selected_indices.append(selected_cat_indices)
            if selected_indices:
                selected_indices = torch.cat(selected_indices)
            else:
                selected_indices = torch.tensor([], dtype=torch.long)
            
            final_boxes = final_boxes[selected_indices].cpu().numpy()
            final_scores = final_scores[selected_indices].cpu().numpy()
            final_labels = final_labels[selected_indices].cpu().numpy()

        # Convert original image to NumPy array for plotting.
        img_np = np.array(pil_img)

        # Create a matplotlib figure.
        fig, ax = plt.subplots(1, figsize=(8, 8))
        ax.imshow(img_np)

        # Generate a dynamic color map based on unique labels.
        unique_labels = np.unique(final_labels)
        n_labels = len(unique_labels)
        cmap = plt.cm.get_cmap('tab20', n_labels)
        color_map = {label: cmap(i) for i, label in enumerate(unique_labels)}
        default_color = (1, 1, 0, 1)  # Yellow in RGBA (if a label is not found)

        # Draw each bounding box.
        for box, score, label in zip(final_boxes, final_scores, final_labels):
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            color = color_map.get(label, default_color)
            rect = patches.Rectangle((x1, y1), width, height,
                                    linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            ax.text(x1, y1, f"Cat {label}: {score:.2f}", color=color,
                    fontsize=8, backgroundcolor='white')

        # Save the image with detections.
        output_path = os.path.join(output_dir, os.path.basename(image_path))
        plt.axis("off")
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    print(f"Inference complete. Processed images saved to {output_dir}.")