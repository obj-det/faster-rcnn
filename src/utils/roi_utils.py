"""
Region of Interest (ROI) utilities for Faster R-CNN.

This module provides utilities for handling ROIs in the Faster R-CNN pipeline:
- ROI score extraction and processing
- Non-maximum suppression (NMS)
- ROI to FPN level mapping
- ROI feature extraction through ROI Align
- Bilinear interpolation for feature sampling
"""

import torch
import math

import torchvision.ops as ops

def get_scores(rpn_output):
    """Extract objectness scores from RPN output.
    
    Reshapes and permutes the RPN output tensor to extract binary classification
    scores (background vs. foreground) for each anchor.
    
    Args:
        rpn_output (torch.Tensor): Raw RPN output of shape [B, S, H, W]
            where S = num_anchors * 2 (2 classes per anchor)
    
    Returns:
        torch.Tensor: Reshaped scores of shape [N, 2] where N = B * H * W * num_anchors
    """
    B, S, H, W = rpn_output.size()
    scores = rpn_output.view(B, -1, 2, H, W)
    scores = scores.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 2)
    return scores.view(-1, 2)

def nms(rois, scores, iou_threshold=0.7, score_threshold=0.05):
    """Apply non-maximum suppression to ROIs.
    
    This function:
    1. Filters ROIs by score threshold
    2. Applies NMS using torchvision's optimized implementation
    
    Args:
        rois (torch.Tensor): ROIs of shape [N, 5] (batch_idx, x1, y1, x2, y2)
        scores (torch.Tensor): Objectness scores of shape [N]
        iou_threshold (float): IoU threshold for NMS. Default: 0.7
        score_threshold (float): Minimum score to keep. Default: 0.05
    
    Returns:
        torch.Tensor: Indices of kept ROIs after NMS
    """
    # Apply score threshold
    keep_initial = scores > score_threshold
    rois = rois[keep_initial]
    scores = scores[keep_initial]
    
    if rois.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=rois.device)

    # Apply NMS
    keep = ops.nms(rois[:, 1:], scores, iou_threshold)
    return keep

def map_rois_to_fpn_levels(rois, k0=4, canonical_scale=224, min_level=3, max_level=5):
    """Map ROIs to appropriate FPN levels based on their scale.
    
    This function implements the paper's ROI level assignment strategy:
    - Compute ROI scale as sqrt(width * height)
    - Assign level using k0 + log2(scale/224)
    - Clamp to valid FPN levels
    
    Args:
        rois (torch.Tensor): ROIs of shape [N, 5] (batch_idx, x1, y1, x2, y2)
        k0 (int): Base FPN level (typically 4). Default: 4
        canonical_scale (float): Reference scale for level assignment. Default: 224
        min_level (int): Minimum FPN level to use. Default: 3
        max_level (int): Maximum FPN level to use. Default: 5
    
    Returns:
        torch.Tensor: Tensor of shape [N] containing the target FPN level for each ROI
    """
    # Extract box coordinates (ignore batch_idx)
    boxes = rois[:, 1:5]
    
    # Compute width and height (add 1 for inclusive coordinates)
    widths  = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    
    # Compute scale as sqrt of area
    scales = torch.sqrt(widths * heights)
    
    # Compute target level using formula from paper
    target_levels = torch.floor(k0 + torch.log2(scales / canonical_scale + 1e-6))
    
    # Clamp to valid FPN levels
    target_levels = torch.clamp(target_levels, min=min_level, max=max_level)
    
    return target_levels.long()

def bilinear_interpolate(feature, x, y):
    """Perform bilinear interpolation on a feature map at given coordinates.
    
    This function:
    1. Finds the four nearest integer coordinates
    2. Retrieves feature values at these points
    3. Computes interpolation weights
    4. Returns weighted sum of features
    
    Args:
        feature (torch.Tensor): Feature map of shape [C, H, W]
        x (float): x-coordinate in feature map space
        y (float): y-coordinate in feature map space
    
    Returns:
        torch.Tensor: Interpolated feature vector of shape [C]
    """
    C, H, W = feature.shape

    # Handle out-of-bounds coordinates
    if x < 0 or x > W - 1 or y < 0 or y > H - 1:
        return torch.zeros(C, device=feature.device, dtype=feature.dtype)

    # Get integer coordinates
    x0 = int(math.floor(x))
    x1 = min(x0 + 1, W - 1)
    y0 = int(math.floor(y))
    y1 = min(y0 + 1, H - 1)

    # Get corner values
    Ia = feature[:, y0, x0]  # Top-left
    Ib = feature[:, y0, x1]  # Top-right
    Ic = feature[:, y1, x0]  # Bottom-left
    Id = feature[:, y1, x1]  # Bottom-right

    # Compute interpolation weights
    wa = (x1 - x) * (y1 - y)  # Top-left weight
    wb = (x - x0) * (y1 - y)  # Top-right weight
    wc = (x1 - x) * (y - y0)  # Bottom-left weight
    wd = (x - x0) * (y - y0)  # Bottom-right weight

    # Return weighted sum
    return Ia * wa + Ib * wb + Ic * wc + Id * wd

def perform_roi_align(levels, all_proposals, fpn_features, pooled_height, pooled_width, img_shape):
    """Perform ROI Align on proposals using appropriate FPN levels.
    
    This function:
    1. Maps each ROI to appropriate FPN level
    2. Applies ROI Align to extract fixed-size features
    3. Handles batching and proper feature map scaling
    
    Args:
        levels (list): List of FPN levels to use
        all_proposals (torch.Tensor): Proposals of shape [N, 5] (batch_idx, x1, y1, x2, y2)
        fpn_features (dict): Dictionary mapping level names to feature maps
        pooled_height (int): Output height for ROI Align
        pooled_width (int): Output width for ROI Align
        img_shape (tuple): Original image dimensions (H, W)
    
    Returns:
        torch.Tensor: ROI-aligned features of shape [N, C, pooled_height, pooled_width]
    """
    # Map ROIs to FPN levels
    assigned_levels = map_rois_to_fpn_levels(all_proposals, min_level=levels[0], max_level=levels[-1])

    N = all_proposals.size(0)
    B, C, H, W = fpn_features['conv' + str(levels[0])].size()

    # Initialize output tensor
    roi_aligned_features = torch.zeros((N, C, pooled_height, pooled_width), device=all_proposals.device)

    # Process each FPN level
    for level in levels:
        # Get ROIs assigned to this level
        level_inds = (assigned_levels == level).nonzero(as_tuple=True)[0]
        if level_inds.numel() == 0:
            continue

        proposals_level = all_proposals[level_inds]
        feature_map = fpn_features['conv' + str(level)]

        # Compute spatial scale for this level
        spatial_scale = feature_map.size(2) / img_shape[0]

        # Apply ROI Align
        aligned_features = ops.roi_align(
            feature_map, 
            proposals_level, 
            output_size=(pooled_height, pooled_width), 
            spatial_scale=spatial_scale,
            aligned=True  # Use aligned corners for better accuracy
        )

        # Store features
        roi_aligned_features[level_inds] = aligned_features

    return roi_aligned_features