import torch
import math

import torchvision.ops as ops

def get_scores(rpn_output):
    B, S, H, W = rpn_output.size()
    scores = rpn_output.view(B, -1, 2, H, W)
    scores = scores.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 2)
    return scores.view(-1, 2)

def nms_old(rois, scores, iou_threshold=0.7, score_threshold=0.05):
    keep_initial = torch.sigmoid(scores) > score_threshold

    rois = rois[keep_initial]
    scores = scores[keep_initial]

    if rois.numel() == 0:
        keep = torch.empty((0,), dtype=torch.int64, device=rois.device)

    x1 = rois[:, 0]
    y1 = rois[:, 1]
    x2 = rois[:, 2]
    y2 = rois[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    _, order = scores.sort(descending=True)
    keep = []

    while order.numel() > 0:
        i = order[0].item()
        keep.append(i)

        if order.numel() == 1:
            break

        # 4. Compute the coordinates for the intersection between the current box and all remaining boxes:
        xx1 = torch.max(x1[i], x1[order[1:]])
        yy1 = torch.max(y1[i], y1[order[1:]])
        xx2 = torch.min(x2[i], x2[order[1:]])
        yy2 = torch.min(y2[i], y2[order[1:]])
        
        # 5. Compute the width and height of the overlapping area:
        w = (xx2 - xx1 + 1).clamp(min=0)
        h = (yy2 - yy1 + 1).clamp(min=0)
        inter = w * h  # Intersection area
        
        # 6. Compute the union area and the Intersection over Union (IoU):
        union = areas[i] + areas[order[1:]] - inter
        iou = inter / union

        # 7. Identify boxes that have an IoU less than or equal to the threshold:
        inds = (iou <= iou_threshold).nonzero(as_tuple=False).squeeze(-1)
        if inds.numel() == 0:
            break  # If no boxes remain with IoU below threshold, exit the loop.
        
        # 8. Update the list of indices (order) to consider for the next iteration.
        # order[0] was the current box, so we only update the remaining ones:
        order = order[inds + 1]

    keep = torch.tensor(keep, dtype=torch.long, device=rois.device)
    return keep

def nms(rois, scores, iou_threshold=0.7, score_threshold=0.05):
    # Apply sigmoid to get probabilities and threshold them.
    keep_initial = torch.sigmoid(scores) > score_threshold
    rois = rois[keep_initial]
    scores = scores[keep_initial]
    
    if rois.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=rois.device)
    
    # Use the built-in, optimized NMS.
    print('='*10)
    print('inside nms')
    print(rois[:5, :])
    print('='*10)
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

def custom_roi_align(feature_map, rois, output_size, spatial_scale=1.0, sampling_ratio=2):
    """
    Custom ROI Align function.

    Parameters:
        feature_map (Tensor): Feature map of shape [B, C, H, W].
        rois (Tensor): ROI tensor of shape [N, 5] where each ROI is [batch_idx, x1, y1, x2, y2].
        output_size (tuple): (pooled_height, pooled_width) for the output.
        spatial_scale (float): Factor to scale ROI coordinates to feature map space.
        sampling_ratio (int): Number of samples per bin dimension. If <=0, adaptive sampling is used.
    
    Returns:
        Tensor: ROI-aligned features of shape [N, C, pooled_height, pooled_width].
    """
    B, C, H, W = feature_map.shape
    N = rois.shape[0]
    pooled_height, pooled_width = output_size

    # Prepare the output tensor.
    output = torch.zeros((N, C, pooled_height, pooled_width), device=feature_map.device, dtype=feature_map.dtype)

    # Process each ROI independently.
    for n in range(N):
        roi = rois[n]
        batch_idx = int(roi[0].item())

        # Scale the ROI coordinates to the feature map space.
        x1 = roi[1] * spatial_scale
        y1 = roi[2] * spatial_scale
        x2 = roi[3] * spatial_scale
        y2 = roi[4] * spatial_scale

        # Compute the ROI's width and height (ensure minimum size of 1).
        roi_width = max(x2 - x1, 1.0)
        roi_height = max(y2 - y1, 1.0)

        # Compute bin sizes
        bin_size_w = roi_width / pooled_width
        bin_size_h = roi_height / pooled_height

        # Loop over each bin in the grid.
        for ph in range(pooled_height):
            for pw in range(pooled_width):
                # Determine the bin boundaries.
                bin_start_w = x1 + pw * bin_size_w
                bin_start_h = y1 + ph * bin_size_h
                bin_end_w = bin_start_w + bin_size_w
                bin_end_h = bin_start_h + bin_size_h

                # Decide how many samples to take. If sampling_ratio > 0, use that.
                if sampling_ratio > 0:
                    num_samples_w = sampling_ratio
                    num_samples_h = sampling_ratio
                else:
                    num_samples_w = math.ceil(bin_size_w)
                    num_samples_h = math.ceil(bin_size_h)

                # Accumulate the interpolated features.
                accumulated_val = torch.zeros(C, device=feature_map.device, dtype=feature_map.dtype)
                count = 0

                for iy in range(num_samples_h):
                    # Compute the y coordinate for this sample.
                    y = bin_start_h + (iy + 0.5) * bin_size_h / num_samples_h
                    for ix in range(num_samples_w):
                        # Compute the x coordinate for this sample.
                        x = bin_start_w + (ix + 0.5) * bin_size_w / num_samples_w
                        val = bilinear_interpolate(feature_map[batch_idx], x, y)
                        accumulated_val += val
                        count += 1

                # The output for this bin is the average of sampled values.
                output[n, :, ph, pw] = accumulated_val / count

    return output

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

        spatial_scale = img_shape[0] / feature_map.size(2)

        aligned_features = custom_roi_align(
            feature_map, 
            proposals_level, 
            output_size=(pooled_height, pooled_width), 
            spatial_scale=spatial_scale, 
            sampling_ratio=2
        )

        roi_aligned_features[level_inds] = aligned_features

    return roi_aligned_features