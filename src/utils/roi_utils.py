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