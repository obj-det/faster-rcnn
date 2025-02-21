import torch
import numpy as np

def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    
    Parameters:
        anchor (ndarray): A 4-element array (x1, y1, x2, y2).
        
    Returns:
        w, h, x_ctr, y_ctr: width, height, x center, and y center of the anchor.
    """
    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr

def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given vectors of widths (ws) and heights (hs) around a center (x_ctr, y_ctr),
    output a set of anchors (windows).
    
    Parameters:
        ws (ndarray): widths, shape (N,)
        hs (ndarray): heights, shape (N,)
        x_ctr (float): x center
        y_ctr (float): y center
        
    Returns:
        anchors (ndarray): Array of shape (N, 4)
    """
    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((
        x_ctr - 0.5 * (ws - 1),
        y_ctr - 0.5 * (hs - 1),
        x_ctr + 0.5 * (ws - 1),
        y_ctr + 0.5 * (hs - 1)
    ))
    return anchors

def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    
    Parameters:
        anchor (ndarray): Base anchor (4,)
        ratios (list): List of aspect ratios
        
    Returns:
        anchors (ndarray): Array of shape (len(ratios), 4)
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / np.array(ratios)
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * np.array(ratios))
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    
    Parameters:
        anchor (ndarray): Anchor (4,)
        scales (list): List of scales
        
    Returns:
        anchors (ndarray): Array of shape (len(scales), 4)
    """
    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * np.array(scales)
    hs = h * np.array(scales)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors

def generate_anchors(base_size=16, ratios=[0.5, 1, 2], scales=[8, 16, 32]):
    """
    Generate base anchors by enumerating aspect ratios and scales.
    
    Parameters:
        base_size (int): The size of the base anchor (typically the stride of the feature map).
        ratios (list): Aspect ratios (height/width) to enumerate.
        scales (list): Anchor scales (multiplicative factors).
        
    Returns:
        anchors (ndarray): Array of shape (N, 4), where N = len(ratios) * len(scales).
                           Each row is (x1, y1, x2, y2).
    """
    # Create a base anchor centered at (0, 0)
    base_anchor = np.array([0, 0, base_size - 1, base_size - 1])
    
    # Enumerate anchors for each aspect ratio
    ratio_anchors = _ratio_enum(base_anchor, ratios)
    
    # For each ratio anchor, enumerate anchors for each scale
    anchors = np.vstack([_scale_enum(ratio_anchors[i, :], scales)
                         for i in range(ratio_anchors.shape[0])])
    
    return anchors

def shift(anchors, feat_height, feat_width, feat_stride):
    shift_x = np.arange(0, feat_width * feat_stride, feat_stride)
    shift_y = np.arange(0, feat_height * feat_stride, feat_stride)

    shift_x, shift_y = np.meshgrid(shift_x, shift_y)

    shifts = np.vstack((
        shift_x.ravel(), 
        shift_y.ravel(),
        shift_x.ravel(), 
        shift_y.ravel()
    )).transpose()

    anchors = anchors.reshape(1, -1, 4) + shifts.reshape(-1, 1, 4)

    all_anchors = anchors.reshape((anchors.shape[0] * anchors.shape[1], 4))

    return all_anchors

def decode_boxes(anchors, deltas, im_shape):
    """
    Decode bounding boxes from anchors and deltas (both of shape [N, 4]) using PyTorch tensors.
    
    Parameters:
        anchors (Tensor): Tensor of shape [N, 4] with anchor boxes in [x1, y1, x2, y2] format.
        deltas (Tensor): Tensor of shape [N, 4] with predicted offsets (dx, dy, dw, dh).
    
    Returns:
        pred_boxes (Tensor): Tensor of shape [N, 4] with final bounding boxes.
    """
    # Compute widths, heights, and center coordinates for the anchors.
    widths  = anchors[:, 2] - anchors[:, 0] + 1.0  # Shape: [N]
    heights = anchors[:, 3] - anchors[:, 1] + 1.0    # Shape: [N]
    ctr_x   = anchors[:, 0] + 0.5 * (widths - 1.0)     # Shape: [N]
    ctr_y   = anchors[:, 1] + 0.5 * (heights - 1.0)    # Shape: [N]
    
    # Extract the deltas for each anchor.
    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]
    
    # Apply the deltas: adjust center coordinates and sizes.
    pred_ctr_x = dx * widths + ctr_x    # New center x
    pred_ctr_y = dy * heights + ctr_y   # New center y
    pred_w = torch.exp(dw) * widths     # New width
    pred_h = torch.exp(dh) * heights    # New height
    
    # Convert center/size back to corner coordinates.
    x1 = torch.clamp(pred_ctr_x - 0.5 * (pred_w - 1.0), 0, im_shape[1] - 1)
    y1 = torch.clamp(pred_ctr_y - 0.5 * (pred_h - 1.0), 0, im_shape[0] - 1)
    x2 = torch.clamp(pred_ctr_x + 0.5 * (pred_w - 1.0), 0, im_shape[1] - 1)
    y2 = torch.clamp(pred_ctr_y + 0.5 * (pred_h - 1.0), 0, im_shape[0] - 1)
    pred_boxes = torch.stack([x1, y1, x2, y2], dim=1)

    # pred_boxes[:, 0] = pred_boxes[:, 0].clamp(0, im_shape[1] - 1)  # x1
    # pred_boxes[:, 1] = pred_boxes[:, 1].clamp(0, im_shape[0] - 1)  # y1
    # pred_boxes[:, 2] = pred_boxes[:, 2].clamp(0, im_shape[1] - 1)  # x2
    # pred_boxes[:, 3] = pred_boxes[:, 3].clamp(0, im_shape[0] - 1)  # y2
    
    return pred_boxes

def collect_rpn_deltas(rpn_deltas):
    B, C, H, W = rpn_deltas.shape
    num_anchors = C // 4
    rpn_deltas = rpn_deltas.view(B, num_anchors, 4, H, W)
    rpn_deltas = rpn_deltas.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4)
    batch_indices = torch.arange(B, dtype=rpn_deltas.dtype, device=rpn_deltas.device).view(B, 1)
    batch_indices = batch_indices.expand(B, rpn_deltas.size(1)).unsqueeze(2)
    rpn_deltas = torch.cat([batch_indices, rpn_deltas], dim=2).view(-1, 5)
    return rpn_deltas

def generate_rois(rpn_deltas, shifted_anchors, im_shape):
    """
    Generate candidate RoIs by applying predicted bounding-box deltas to the anchors.
    
    Parameters:
        feature_map (Tensor): Feature map from the backbone network.
        rpn_deltas (Tensor): Predicted bounding-box deltas from the RPN.
        anchors (Tensor): Anchor boxes in [x1, y1, x2, y2] format.
        im_shape (tuple): Shape of the input image (H x W).
        
    Returns:
        rois (Tensor): Region of interests (RoIs) in [x1, y1, x2, y2] format.
    """
    im_h, im_w = im_shape
    B, C, H, W = rpn_deltas.shape

    num_anchors = C // 4

    rpn_deltas = rpn_deltas.view(B, num_anchors, 4, H, W)

    rpn_deltas = rpn_deltas.permute(0, 3, 4, 1, 2).contiguous().view(B, -1, 4)
    # anchors = shift(anchors, H, W, im_h // H)

    # anchors = torch.from_numpy(anchors).to(rpn_deltas.device).float()

    rois = torch.stack([decode_boxes(shifted_anchors, rpn_deltas[i], (im_h, im_w)) for i in range(B)], dim=0)

    batch_indices = torch.arange(B, dtype=rois.dtype, device=rois.device).view(B, 1)

    batch_indices = batch_indices.expand(B, rois.size(1)).unsqueeze(2)

    rois = torch.cat([batch_indices, rois], dim=2).view(-1, 5)

    return rois