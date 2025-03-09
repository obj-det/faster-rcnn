"""
Anchor box utilities for Faster R-CNN.

This module provides utilities for handling anchor boxes:
- Anchor box generation with different scales and aspect ratios
- Anchor box transformation and decoding
- ROI generation from RPN outputs
- Helper functions for anchor box manipulation

The anchor box system is a key component of Faster R-CNN, providing the initial
set of candidate object locations that are refined by the network.
"""

import torch
import numpy as np

def _whctrs(anchor):
    """Convert anchor box coordinates to width, height, and center coordinates.
    
    Args:
        anchor (numpy.ndarray): Anchor box coordinates [x1, y1, x2, y2]
    
    Returns:
        tuple: (width, height, x_center, y_center) of the anchor box
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """Create anchor boxes from widths, heights, and center coordinates.
    
    Args:
        ws (numpy.ndarray): Anchor widths of shape (N,)
        hs (numpy.ndarray): Anchor heights of shape (N,)
        x_ctr (float): x-coordinate of anchor center
        y_ctr (float): y-coordinate of anchor center
    
    Returns:
        numpy.ndarray: Anchor boxes of shape (N, 4) in [x1, y1, x2, y2] format
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((
        x_ctr - 0.5 * (ws - 1),  # x1
        y_ctr - 0.5 * (hs - 1),  # y1
        x_ctr + 0.5 * (ws - 1),  # x2
        y_ctr + 0.5 * (hs - 1)   # y2
    ))
    return anchors

def _ratio_enum(anchor, ratios):
    """Enumerate anchor boxes with different aspect ratios.
    
    For a given base anchor, generates a set of new anchors by varying the
    aspect ratio while keeping the area constant.
    
    Args:
        anchor (numpy.ndarray): Base anchor box [x1, y1, x2, y2]
        ratios (list): List of aspect ratios (height/width)
    
    Returns:
        numpy.ndarray: Generated anchor boxes of shape (len(ratios), 4)
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / np.array(ratios)
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * np.array(ratios))
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """Enumerate anchor boxes with different scales.
    
    For a given base anchor, generates a set of new anchors by scaling
    both width and height while maintaining the aspect ratio.
    
    Args:
        anchor (numpy.ndarray): Base anchor box [x1, y1, x2, y2]
        scales (list): List of scale factors
    
    Returns:
        numpy.ndarray: Generated anchor boxes of shape (len(scales), 4)
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * np.array(scales)
    hs = h * np.array(scales)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """Generate a complete set of anchor boxes.
    
    This is the main anchor generation function that combines aspect ratio
    and scale enumeration to create a diverse set of anchor boxes.
    
    Args:
        base_size (int): Base size of anchors (typically feature stride)
        ratios (list): Aspect ratios (height/width) for anchors
        scales (list): Scale factors for anchors
    
    Returns:
        numpy.ndarray: Generated anchor boxes of shape (N, 4) where
                      N = len(ratios) * len(scales)
    """
    # Create base anchor at origin
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])
    
    # Generate anchors with different aspect ratios
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    
    # For each aspect ratio, generate anchors with different scales
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    
    return anchors

def shift(anchors, feat_height, feat_width, feat_stride):
    """Shift anchor boxes across feature map positions.
    
    Creates a complete set of anchor boxes by shifting the base anchors to
    every position in the feature map, taking into account the feature stride.
    
    Args:
        anchors (numpy.ndarray): Base anchor boxes
        feat_height (int): Height of feature map
        feat_width (int): Width of feature map
        feat_stride (int): Stride of feature map relative to input image
    
    Returns:
        numpy.ndarray: Complete set of anchor boxes for the feature map
    """
    # Generate shift coordinates
    shift_x = np.arange(0, feat_width * feat_stride, feat_stride)
    shift_y = np.arange(0, feat_height * feat_stride, feat_stride)
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    # Stack shifts into [dx, dy, dx, dy] format
    shifts = np.vstack((
        shift_x.ravel(), 
        shift_y.ravel(),
        shift_x.ravel(), 
        shift_y.ravel()
    )).transpose()

    # Add shifts to base anchors
    anchors = anchors.reshape(1, -1, 4) + shifts.reshape(-1, 1, 4)
    all_anchors = anchors.reshape((anchors.shape[0] * anchors.shape[1], 4))

    return all_anchors

def decode_boxes(anchors, deltas, im_shape):
    """Decode predicted box deltas relative to anchor boxes.
    
    Converts the predicted box deltas from the RPN into actual box coordinates
    using the anchor boxes as reference. Also handles clipping to image boundaries.
    
    Args:
        anchors (torch.Tensor): Anchor boxes of shape [N, 4] in [x1, y1, x2, y2] format
        deltas (torch.Tensor): Predicted deltas of shape [N, 4] in [dx, dy, dw, dh] format
        im_shape (tuple): Image dimensions (height, width)
    
    Returns:
        torch.Tensor: Decoded boxes of shape [N, 4] in [x1, y1, x2, y2] format,
                     clipped to image boundaries
    """
    # Extract anchor box dimensions
    widths = anchors[:, 2] - anchors[:, 0] + 1.0
    heights = anchors[:, 3] - anchors[:, 1] + 1.0
    ctr_x = anchors[:, 0] + 0.5 * (widths - 1.0)
    ctr_y = anchors[:, 1] + 0.5 * (heights - 1.0)
    
    # Extract predicted deltas
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    
    # Apply deltas to anchors
    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = torch.exp(dw) * widths
    pred_h = torch.exp(dh) * heights
    
    # Convert to corner coordinates and clip to image boundaries
    x1 = torch.clamp(pred_ctr_x - 0.5 * (pred_w - 1.0), 0, im_shape[1] - 1)
    y1 = torch.clamp(pred_ctr_y - 0.5 * (pred_h - 1.0), 0, im_shape[0] - 1)
    x2 = torch.clamp(pred_ctr_x + 0.5 * (pred_w - 1.0), 0, im_shape[1] - 1)
    y2 = torch.clamp(pred_ctr_y + 0.5 * (pred_h - 1.0), 0, im_shape[0] - 1)
    
    pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)
    return pred_boxes

def collect_rpn_deltas(rpn_deltas):
    """Reshape and reorganize RPN delta predictions.
    
    Converts RPN delta predictions from feature map format to a batch of
    predictions with batch indices.
    
    Args:
        rpn_deltas (torch.Tensor): RPN delta predictions of shape [B, C, H, W]
                                  where C = num_anchors * 4
    
    Returns:
        torch.Tensor: Reshaped deltas of shape [B*H*W*num_anchors, 5]
                     where each row is [batch_idx, dx, dy, dw, dh]
    """
    B, C, H, W = rpn_deltas.shape
    num_anchors = C // 4
    rpn_deltas = rpn_deltas.view(B, num_anchors, 4, H, W)
    rpn_deltas = rpn_deltas.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4)
    batch_indices = torch.arange(B, dtype=rpn_deltas.dtype, device=rpn_deltas.device).view(B, 1)
    batch_indices = batch_indices.expand(B, rpn_deltas.size(1)).unsqueeze(2)
    rpn_deltas = torch.cat([batch_indices, rpn_deltas], dim=2).view(-1, 5)
    return rpn_deltas

def generate_rois(rpn_deltas, shifted_anchors, im_shape):
    """Generate Region of Interest (ROI) proposals from RPN predictions.
    
    This function:
    1. Reshapes RPN delta predictions
    2. Applies deltas to shifted anchor boxes
    3. Adds batch indices to proposals
    
    Args:
        rpn_deltas (torch.Tensor): RPN delta predictions [B, C, H, W]
        shifted_anchors (torch.Tensor): Shifted anchor boxes
        im_shape (tuple): Image dimensions (height, width)
    
    Returns:
        torch.Tensor: ROI proposals of shape [B*H*W*num_anchors, 5]
                     where each row is [batch_idx, x1, y1, x2, y2]
    """
    im_h, im_w = im_shape
    B, C, H, W = rpn_deltas.shape
    num_anchors = C // 4

    # Reshape deltas for processing
    rpn_deltas = rpn_deltas.view(B, num_anchors, 4, H, W)
    rpn_deltas = rpn_deltas.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4)

    # Generate ROIs for each batch
    rois = torch.stack([decode_boxes(shifted_anchors, rpn_deltas[i], (im_h, im_w)) 
                       for i in range(B)], dim=0)

    # Add batch indices
    batch_indices = torch.arange(B, dtype=rois.dtype, device=rois.device).view(B, 1)
    batch_indices = batch_indices.expand(B, rois.size(1)).unsqueeze(2)
    rois = torch.cat([batch_indices, rois], dim=2).view(-1, 5)

    return rois