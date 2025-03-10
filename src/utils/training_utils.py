"""
Training utilities for Faster R-CNN object detection.

This module provides utilities for training the Faster R-CNN model:
- Loss functions for RPN and detection head
- IoU computation and bounding box transformations
- Proposal matching and sampling strategies
- Training and validation loops with metric tracking
- Batch processing utilities

The training process follows these key steps:
1. Generate proposals using RPN
2. Match proposals to ground truth boxes
3. Sample positive and negative proposals
4. Compute regression targets
5. Train RPN and detection head with appropriate losses
"""

from tqdm import tqdm
import torch
import torch.nn.functional as F
from pycocotools.coco import COCO
import tempfile
import json
import os
from torchvision.transforms import ToPILImage

from src.utils.anchor_utils import *
from src.utils.roi_utils import *
from src.utils.eval_utils import *

def compute_iou(box, boxes):
    """
    Compute IoU between a single box and a set of boxes.
    
    Args:
        box (Tensor): Tensor of shape [4] for a single box: [x1, y1, x2, y2].
        boxes (Tensor): Tensor of shape [K, 4] for K boxes.
        
    Returns:
        iou (Tensor): Tensor of shape [K] containing IoU values.
    """
    # Intersection coordinates
    x1 = torch.max(box[0], boxes[:, 0])
    y1 = torch.max(box[1], boxes[:, 1])
    x2 = torch.min(box[2], boxes[:, 2])
    y2 = torch.min(box[3], boxes[:, 3])
    
    inter_w = (x2 - x1 + 1).clamp(min=0)
    inter_h = (y2 - y1 + 1).clamp(min=0)
    inter = inter_w * inter_h

    # Areas of boxes
    area_box = (box[2] - box[0] + 1) * (box[3] - box[1] + 1)
    areas_boxes = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    
    union = area_box + areas_boxes - inter
    iou = inter / union
    return iou

def bbox_transform(proposals, gt_boxes):
    """
    Compute bounding box regression targets for proposals relative to ground-truth boxes.
    
    Args:
        proposals (Tensor): Tensor of shape [P, 4].
        gt_boxes (Tensor): Tensor of shape [P, 4] (each proposal's matching GT).
        
    Returns:
        targets (Tensor): Tensor of shape [P, 4] containing (dx, dy, dw, dh).
    """
    widths  = proposals[:, 2] - proposals[:, 0] + 1.0
    heights = proposals[:, 3] - proposals[:, 1] + 1.0
    ctr_x   = proposals[:, 0] + 0.5 * widths
    ctr_y   = proposals[:, 1] + 0.5 * heights

    gt_widths  = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_ctr_x   = gt_boxes[:, 0] + 0.5 * gt_widths
    gt_ctr_y   = gt_boxes[:, 1] + 0.5 * gt_heights

    dx = (gt_ctr_x - ctr_x) / widths
    dy = (gt_ctr_y - ctr_y) / heights
    dw = torch.log(gt_widths / widths)
    dh = torch.log(gt_heights / heights)
    
    targets = torch.stack((dx, dy, dw, dh), dim=1)
    return targets

def compute_iou_vectorized(proposals, gt_boxes):
    """
    Compute the IoU between each proposal and each ground truth box in a vectorized manner.

    Args:
        proposals (Tensor): Tensor of shape [N, 4] for proposals.
        gt_boxes (Tensor): Tensor of shape [M, 4] for ground-truth boxes.

    Returns:
        ious (Tensor): IoU matrix of shape [N, M].
    """
    N = proposals.size(0)
    M = gt_boxes.size(0)

    # proposals: [N,4] and gt_boxes: [M,4]
    # Expand dimensions for broadcasting.
    proposals = proposals.unsqueeze(1)  # shape: [N, 1, 4]
    gt_boxes = gt_boxes.unsqueeze(0)      # shape: [1, M, 4]

    # Compute intersection coordinates.
    x1 = torch.max(proposals[..., 0], gt_boxes[..., 0])  # [N, M]
    y1 = torch.max(proposals[..., 1], gt_boxes[..., 1])
    x2 = torch.min(proposals[..., 2], gt_boxes[..., 2])
    y2 = torch.min(proposals[..., 3], gt_boxes[..., 3])

    inter_w = (x2 - x1 + 1).clamp(min=0)
    inter_h = (y2 - y1 + 1).clamp(min=0)
    inter_area = inter_w * inter_h  # [N, M]

    # Compute areas of proposals and gt_boxes.
    proposal_area = ((proposals[..., 2] - proposals[..., 0] + 1) *
                     (proposals[..., 3] - proposals[..., 1] + 1))  # [N, 1]
    gt_area = ((gt_boxes[..., 2] - gt_boxes[..., 0] + 1) *
               (gt_boxes[..., 3] - gt_boxes[..., 1] + 1))         # [1, M]

    union_area = proposal_area + gt_area - inter_area
    ious = inter_area / union_area
    return ious  # shape: [N, M]

def matching_and_sampling(all_proposals, all_scores, gt, num_samples, pos_iou_thresh=0.5, neg_iou_thresh=0.1, pos_fraction=0.25):
    """Match proposals to ground truth boxes and sample for training.
    
    This function implements the proposal matching and sampling strategy:
    1. For each batch, match proposals to ground truth boxes using IoU
    2. Label proposals as positive if IoU > pos_iou_thresh
    3. Label proposals as negative if IoU < neg_iou_thresh
    4. Sample a balanced set of positive and negative proposals
    5. Compute regression targets for positive proposals
    
    Args:
        all_proposals (torch.Tensor): Proposals from RPN [N, 5] (batch_idx, x1, y1, x2, y2)
        all_scores (torch.Tensor): Objectness scores from RPN [N, 2]
        gt (torch.Tensor): Ground truth boxes and labels [M, 6] (batch_idx, x1, y1, x2, y2, label)
        num_samples (int): Total number of proposals to sample per batch
        pos_iou_thresh (float): IoU threshold for positive proposals (default: 0.5)
        neg_iou_thresh (float): IoU threshold for negative proposals (default: 0.1)
        pos_fraction (float): Target fraction of positive samples (default: 0.25)
    
    Returns:
        tuple: (sampled_proposals, sampled_scores, sampled_labels, sampled_bbox_targets, sampled_indices)
            - sampled_proposals: Selected proposals [K, 5]
            - sampled_scores: Objectness scores for selected proposals [K, 2]
            - sampled_labels: Classification labels for selected proposals [K]
            - sampled_bbox_targets: Regression targets for selected proposals [K, 4]
            - sampled_indices: Original indices of selected proposals [K]
    """
    sampled_proposals_list = []
    batch_scores_list = []
    sampled_labels_list = []
    sampled_bbox_targets_list = []
    sampled_indices_list = []

    batch_ids = all_proposals[:, 0].unique()

    for batch_idx in batch_ids:
        batch_mask = all_proposals[:, 0] == batch_idx

        global_indices = torch.nonzero(batch_mask).squeeze(1)
        batch_proposals = all_proposals[batch_mask][:, 1:]
        batch_scores = all_scores[batch_mask]

        gt_mask = gt[:, 0] == batch_idx
        gt_b = gt[gt_mask][:, 1:5]
        gt_labels = gt[gt_mask][:, 5].long()

        N_b = batch_proposals.shape[0]
        M_b = gt_b.shape[0]

        if M_b == 0:
            max_ious = torch.zeros(N_b, device=all_proposals.device)
            gt_assignment = torch.zeros(N_b, dtype=torch.long, device=all_proposals.device)
        else:
            ious = compute_iou_vectorized(batch_proposals, gt_b)  # shape [N_b, M_b]
            max_ious, gt_assignment = ious.max(dim=1)

        # Initialize labels for proposals (default background: label 0).
        labels_b = torch.zeros(N_b, dtype=torch.long, device=all_proposals.device)
        # Regression targets are zeros by default.
        bbox_targets_b = torch.zeros((N_b, 4), device=all_proposals.device)

        pos_inds = max_ious >= pos_iou_thresh
        if pos_inds.sum() > 0:
            labels_b[pos_inds] = gt_labels[gt_assignment[pos_inds]]
            pos_proposals = batch_proposals[pos_inds]
            matched_gt = gt_b[gt_assignment[pos_inds]]
            bbox_targets_b[pos_inds] = bbox_transform(pos_proposals, matched_gt)
        
        valid_inds = (max_ious >= pos_iou_thresh) | (max_ious < neg_iou_thresh)

        num_pos = int(pos_fraction * num_samples)
        pos_idx = torch.nonzero(pos_inds).squeeze(1)
        neg_idx = torch.nonzero((~pos_inds) & valid_inds).squeeze(1)

        num_pos_sample = min(num_pos, pos_idx.numel())
        num_neg_sample = num_samples - num_pos_sample

        if pos_idx.numel() > 0:
            perm_pos = pos_idx[torch.randperm(pos_idx.numel())][:num_pos_sample]
        else:
            perm_pos = pos_idx
        if neg_idx.numel() > 0:
            perm_neg = neg_idx[torch.randperm(neg_idx.numel())][:num_neg_sample]
        else:
            perm_neg = neg_idx
        
        keep_inds = torch.cat([perm_pos, perm_neg], dim=0)
        
        sampled_batch_indices = global_indices[keep_inds]
        sampled_indices_list.append(sampled_batch_indices)

        proposals_keep = batch_proposals[keep_inds]  # shape [K_b, 4]
        batch_scores_keep = batch_scores[keep_inds]  # shape [K_b]
        labels_keep = labels_b[keep_inds]         # shape [K_b]
        bbox_targets_keep = bbox_targets_b[keep_inds]  # shape [K_b, 4]
        
        batch_idx_tensor = torch.full((proposals_keep.shape[0], 1), batch_idx.item(), device=all_proposals.device)
        proposals_keep = torch.cat([batch_idx_tensor, proposals_keep], dim=1)  # now [K_b, 5]

        sampled_proposals_list.append(proposals_keep)
        batch_scores_list.append(batch_scores_keep)
        sampled_labels_list.append(labels_keep)
        sampled_bbox_targets_list.append(bbox_targets_keep)

    # Concatenate results from all batches.
    sampled_proposals = torch.cat(sampled_proposals_list, dim=0) if sampled_proposals_list else torch.empty(0, 5, dtype=torch.float32, device=all_proposals.device)
    sampled_scores = torch.cat(batch_scores_list, dim=0) if batch_scores_list else torch.empty(0, 2, dtype=torch.float32, device=all_proposals.device)
    sampled_labels = torch.cat(sampled_labels_list, dim=0) if sampled_labels_list else torch.empty(0, dtype=torch.int64, device=all_proposals.device)
    sampled_bbox_targets = torch.cat(sampled_bbox_targets_list, dim=0) if sampled_bbox_targets_list else torch.empty(0, 4, dtype=torch.float32, device=all_proposals.device)
    sampled_indices = torch.cat(sampled_indices_list, dim=0) if sampled_indices_list else torch.empty(0, dtype=torch.int64, device=all_proposals.device)

    return sampled_proposals, sampled_scores, sampled_labels, sampled_bbox_targets, sampled_indices

def add_batch_idx_to_targets(targets):
    """
    Add batch index as the first column to each target's bounding boxes.

    Args:
        targets (list of dict): Each dict has at least the key "boxes", a tensor of shape [num_boxes, 4].

    Returns:
        list of dict: A new list of target dictionaries with "boxes" of shape [num_boxes, 5],
                      where the first column is the batch index.
    """
    new_targets = [
        {**target, "boxes_with_label": torch.cat([
            torch.full((target["boxes_with_label"].shape[0], 1),
                       i,
                       dtype=target["boxes_with_label"].dtype,
                       device=target["boxes_with_label"].device),
            target["boxes_with_label"]
        ], dim=1)}
        for i, target in enumerate(targets)
    ]
    return new_targets

def rpn_loss_fn(sampled_deltas, sampled_scores, sampled_labels, sampled_bbox_targets):
    """Compute RPN losses for classification and box regression.
    
    Args:
        sampled_deltas (torch.Tensor): Predicted box deltas [N, 5] (batch_idx, dx, dy, dw, dh)
        sampled_scores (torch.Tensor): Predicted objectness scores [N, 2]
        sampled_labels (torch.Tensor): Ground truth labels [N] (0: background, 1: foreground)
        sampled_bbox_targets (torch.Tensor): Ground truth box deltas [N, 4]
    
    Returns:
        tuple: (cls_loss, reg_loss)
            - cls_loss: Binary cross-entropy loss for objectness prediction
            - reg_loss: Smooth L1 loss for box delta prediction (only for positive anchors)
    """
    cls_loss = F.cross_entropy(sampled_scores, sampled_labels)
    pos_inds = (sampled_labels == 1).nonzero(as_tuple=True)[0]

    if pos_inds.numel() > 0:
        # Use Smooth L1 loss on positive anchors.
        loss_reg = F.smooth_l1_loss(sampled_deltas[pos_inds, 1:],
                                    sampled_bbox_targets[pos_inds],
                                    reduction='sum')
        # Normalize by the number of positive anchors.
        loss_reg = loss_reg / pos_inds.numel()
    else:
        loss_reg = torch.tensor(0.0, device=sampled_scores.device)
    
    return cls_loss, loss_reg

def det_loss_fn(sampled_bbox_preds, sampled_scores, sampled_labels, sampled_bbox_targets, num_classes):
    """Compute detection head losses for classification and box regression.
    
    Args:
        sampled_bbox_preds (torch.Tensor): Predicted box deltas [N, num_classes * 4]
        sampled_scores (torch.Tensor): Predicted class scores [N, num_classes + 1]
        sampled_labels (torch.Tensor): Ground truth class labels [N]
        sampled_bbox_targets (torch.Tensor): Ground truth box deltas [N, 4]
        num_classes (int): Number of object classes (excluding background)
    
    Returns:
        tuple: (cls_loss, reg_loss)
            - cls_loss: Cross-entropy loss for class prediction
            - reg_loss: Smooth L1 loss for box delta prediction (only for positive samples)
    """
    if sampled_scores.size(0) == 0:
        cls_loss = torch.tensor(0.0, device=sampled_scores.device)
    else:
        cls_loss = F.cross_entropy(sampled_scores, sampled_labels)
    
    pos_inds = torch.nonzero(sampled_labels > 0).squeeze(1)

    if pos_inds.numel() > 0:
        sampled_bbox_preds = sampled_bbox_preds.view(sampled_bbox_preds.size(0), num_classes, 4)
        pos_labels = (sampled_labels[pos_inds] - 1).view(-1, 1, 1).expand(-1, 1, 4)
        sampled_bbox_pred_pos = sampled_bbox_preds[pos_inds].gather(1, pos_labels).squeeze(1)

        # Use Smooth L1 loss on positive anchors.
        loss_reg = F.smooth_l1_loss(sampled_bbox_pred_pos,
                                    sampled_bbox_targets[pos_inds],
                                    reduction='sum')
        # Normalize by the number of positive anchors.
        loss_reg = loss_reg / pos_inds.numel()
    else:
        loss_reg = torch.tensor(0.0, device=sampled_scores.device)
    
    return cls_loss, loss_reg

def train_loop(epoch_num, accum_steps, dataloader, backbone, fpn, rpn, head, optimizer, device,
               layer_to_shifted_anchors, img_shape, num_classes, pooled_height, pooled_width, train_logger, config):
    """Execute one epoch of training for the Faster R-CNN model.
    
    This function:
    1. Processes batches of images and targets
    2. Generates region proposals using RPN
    3. Matches proposals to ground truth boxes
    4. Computes RPN and detection head losses
    5. Updates model parameters with gradient accumulation
    6. Logs training metrics
    
    Args:
        epoch_num (int): Current epoch number
        accum_steps (int): Number of steps for gradient accumulation
        dataloader (DataLoader): Training data loader
        backbone (nn.Module): Backbone network for feature extraction
        fpn (nn.Module): Feature Pyramid Network
        rpn (nn.Module): Region Proposal Network
        head (nn.Module): Detection head for classification and box regression
        optimizer (torch.optim.Optimizer): Model optimizer
        device (torch.device): Device to run training on
        layer_to_shifted_anchors (dict): Pre-computed anchors for each FPN level
        img_shape (tuple): Image dimensions (height, width)
        num_classes (int): Number of object classes (excluding background)
        pooled_height (int): Height of ROI features after ROI Align
        pooled_width (int): Width of ROI features after ROI Align
        train_logger (logging.Logger): Logger for training metrics
        config (dict): Configuration dictionary containing training parameters
    """
    backbone.train()
    fpn.train()
    rpn.train()
    head.train()

    leftover_count = len(dataloader) % accum_steps

    optimizer.zero_grad()

    for i, (images, targets) in enumerate(tqdm(dataloader, desc="Training")):
        # Move images and targets to the device
        images = images.to(device)
        targets = add_batch_idx_to_targets(targets)
        # print(images.size(), targets[0]['boxes_with_label'].size(), flush=True)
        gt = torch.cat([t['boxes_with_label'] for t in targets], dim=0).to(device)
        rpn_gt = gt.clone()
        # Convert labels to binary for the RPN (foreground vs background)
        rpn_gt[:, 5] = 1

        # Forward pass: Backbone -> FPN.
        features = backbone(images)
        fpn_features = fpn(features)

        # Forward pass through the RPN at each FPN level.
        rpn_out = {}
        for k, feature_map in fpn_features.items():
            rpn_out[k] = rpn(feature_map)
        sorted_rpn_keys = sorted(rpn_out.keys())

        # Aggregate RPN outputs.
        all_rpn_deltas = torch.cat(
            [collect_rpn_deltas(rpn_out[k][1]) for k in sorted_rpn_keys], dim=0
        )
        rois = torch.cat(
            [generate_rois(rpn_out[k][1], layer_to_shifted_anchors[k], img_shape)
                for k in sorted_rpn_keys], dim=0
        )
        scores = torch.cat(
            [get_scores(rpn_out[k][0]) for k in sorted_rpn_keys], dim=0
        )

        # Matching and sampling for RPN training.
        (rpn_sampled_proposals, rpn_sampled_scores, rpn_sampled_labels,
            rpn_sampled_bbox_targets, rpn_sampled_indices) = matching_and_sampling(
                rois, scores, rpn_gt, config['rpn']['rpn_train_batch_size'], pos_iou_thresh=config['rpn']['rpn_pos_iou_thresh'], neg_iou_thresh=config['rpn']['rpn_neg_iou_thresh'], pos_fraction=config['rpn']['rpn_pos_ratio']
            )
        rpn_sampled_deltas = all_rpn_deltas[rpn_sampled_indices]

        # Process proposals for the detection head.
        unique_batches = rois[:, 0].unique()
        all_proposals = []
        all_scores = []
        for batch_idx in unique_batches:
            batch_mask = rois[:, 0] == batch_idx
            batch_rois = rois[batch_mask]
            batch_scores = scores[batch_mask]
            batch_scores = F.softmax(batch_scores, dim=1)
            
            _, pre_nms_sorted_indices = torch.sort(batch_scores[:, 1], descending=True)
            batch_rois = batch_rois[pre_nms_sorted_indices]
            batch_scores = batch_scores[pre_nms_sorted_indices]

            batch_rois = batch_rois[:config['rpn']['pre_nms_topk_train']]
            batch_scores = batch_scores[:config['rpn']['pre_nms_topk_train']]
            
            keep = nms(batch_rois, batch_scores[:, 1], iou_threshold=config['rpn']['nms_iou_thresh'], score_threshold=config['rpn']['nms_score_thresh'])
            batch_rois = batch_rois[keep]
            batch_scores = batch_scores[keep]

            # Sort proposals by score.
            _, sorted_indices = torch.sort(batch_scores[:, 1], descending=True)
            sorted_proposals = batch_rois[sorted_indices]
            sorted_scores = batch_scores[sorted_indices]

            K = config['rpn']['nms_topk_train']  # Adjust K based on your requirements.
            topk_proposals = sorted_proposals[:K]
            topk_scores = sorted_scores[:K]

            all_proposals.append(topk_proposals)
            all_scores.append(topk_scores)

        all_proposals = torch.cat(all_proposals, dim=0).detach()
        all_scores = torch.cat(all_scores, dim=0).detach()

        # Matching and sampling for the detection head.
        (sampled_proposals, sampled_scores, sampled_labels,
            sampled_bbox_targets, _) = matching_and_sampling(
                all_proposals, all_scores, gt, config['head']['detection_train_batch_size'], pos_iou_thresh=config['head']['detection_pos_iou_thresh'], neg_iou_thresh=config['head']['detection_neg_iou_thresh'], pos_fraction=config['head']['detection_pos_ratio']
        
        )

        # ROI Align: pool features corresponding to the proposals.
        levels = sorted([int(x[-1]) for x in rpn_out.keys()])
        aligned_proposals = perform_roi_align(
            levels, sampled_proposals, fpn_features,
            pooled_height, pooled_width, img_shape
        )

        # Forward pass through the detection head.
        cls_scores, bbox_deltas = head(aligned_proposals)

        # Compute losses.
        rpn_cls_loss, rpn_bbox_loss = rpn_loss_fn(
            rpn_sampled_deltas, rpn_sampled_scores,
            rpn_sampled_labels, rpn_sampled_bbox_targets
        )
        det_cls_loss, det_bbox_loss = det_loss_fn(
            bbox_deltas, cls_scores,
            sampled_labels, sampled_bbox_targets, num_classes
        )
        total_loss = rpn_cls_loss + rpn_bbox_loss + det_cls_loss + det_bbox_loss

        # Accumulate gradients.
        if len(dataloader) - i <= leftover_count:
            total_loss = total_loss / leftover_count
        else:
            total_loss = total_loss / accum_steps
        
        total_loss.backward()

        # Backpropagation and parameter update.
        if (i+1) % accum_steps == 0 or i == len(dataloader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        # Build log message
        log_message = (
            f"Epoch: {epoch_num}, Iteration: {i+1}, Total Loss: {total_loss.item():.4f}, "
            f"RPN_cls: {rpn_cls_loss.item():.4f}, RPN_bbox: {rpn_bbox_loss.item():.4f}, "
            f"DET_cls: {det_cls_loss.item():.4f}, DET_bbox: {det_bbox_loss.item():.4f}"
        )
        # Log to file
        train_logger.info(log_message)

def validation_loop(epoch_num, dataloader, backbone, fpn, rpn, head, device,
                   layer_to_shifted_anchors, img_shape, num_classes,
                   pooled_height, pooled_width, val_logger, config, ap_iou_thresholds):
    """Execute one epoch of validation for the Faster R-CNN model.
    
    This function:
    1. Processes validation batches in evaluation mode
    2. Generates and filters region proposals
    3. Computes detection scores and box coordinates
    4. Accumulates predictions for COCO evaluation
    5. Computes and logs AP metrics at different IoU thresholds
    
    Args:
        epoch_num (int): Current epoch number
        dataloader (DataLoader): Validation data loader
        backbone (nn.Module): Backbone network for feature extraction
        fpn (nn.Module): Feature Pyramid Network
        rpn (nn.Module): Region Proposal Network
        head (nn.Module): Detection head for classification and box regression
        device (torch.device): Device to run validation on
        layer_to_shifted_anchors (dict): Pre-computed anchors for each FPN level
        img_shape (tuple): Image dimensions (height, width)
        num_classes (int): Number of object classes (excluding background)
        pooled_height (int): Height of ROI features after ROI Align
        pooled_width (int): Width of ROI features after ROI Align
        val_logger (logging.Logger): Logger for validation metrics
        config (dict): Configuration dictionary containing validation parameters
        ap_iou_thresholds (list): IoU thresholds for computing AP metrics
    
    Returns:
        dict: Dictionary containing AP metrics for each IoU threshold
    """
    backbone.eval()
    fpn.eval()
    rpn.eval()
    head.eval()

    # Initialize the evaluator (we'll fill in the ground truth later)
    coco_evaluator = CocoEvaluator(None)

    # Prepare COCO-style ground truth annotations.
    annotations = []
    images = []
    categories = [{"id": i, "name": f"class_{i}"} for i in range(1, num_classes + 1)]
    ann_id = 0
    image_id_map = {}  # Maps dataset index to COCO image_id

    # Disable gradient computation.
    with torch.no_grad():
        for i, (image_batch, targets) in enumerate(tqdm(dataloader, desc="Validation")):
            # Move images to device.
            image_batch = image_batch.to(device)
            targets = add_batch_idx_to_targets(targets)

            # Process ground truth to build COCO annotations.
            for batch_idx, target in enumerate(targets):
                img_idx = i * image_batch.shape[0] + batch_idx
                if img_idx not in image_id_map:
                    image_id = len(image_id_map) + 1  # COCO image IDs start from 1.
                    image_id_map[img_idx] = image_id
                    # image_pil = ToPILImage()(image_batch[batch_idx].cpu())
                    # # Convert to in-memory bytes
                    # import base64
                    # from io import BytesIO
                    # buffer = BytesIO()
                    # image_pil.save(buffer, format="PNG")
                    # byte_data = buffer.getvalue()

                    # # Encode as base64 for JSON
                    # base64_str = base64.b64encode(byte_data).decode("utf-8")
                    images.append({
                        "id": image_id,
                        # "image": base64_str,
                        "width": img_shape[1],
                        "height": img_shape[0]
                    })
                else:
                    image_id = image_id_map[img_idx]
                boxes = target['boxes_with_label'].cpu()
                # original_boxes = target['original_bboxes'].cpu()
                # aug_boxes = target['aug_bboxes'].cpu()
                for box_idx in range(boxes.shape[0]):
                    # orig_x1, orig_y1, orig_x2, orig_y2 = original_boxes[box_idx].tolist()
                    # aug_x1, aug_y1, aug_x2, aug_y2 = aug_boxes[box_idx].tolist()
                    x1, y1, x2, y2 = boxes[box_idx, 1:5].tolist()
                    width = x2 - x1
                    height = y2 - y1
                    category_id = int(boxes[box_idx, 5].item())
                    ann_id += 1
                    annotations.append({
                        "id": ann_id,
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": [x1, y1, width, height],
                        # "original_bbox": [orig_x1, orig_y1, orig_x2-orig_x1, orig_y2-orig_y1],
                        # "aug_bbox": [aug_x1, aug_y1, aug_x2-aug_x1, aug_y2-aug_y1],
                        "area": width * height,
                        "iscrowd": 0
                    })

            # Forward pass: Backbone -> FPN.
            features = backbone(image_batch)
            fpn_features = fpn(features)

            # Run the RPN over each FPN level.
            rpn_out = {}
            for k, feature_map in fpn_features.items():
                rpn_out[k] = rpn(feature_map)
            sorted_rpn_keys = sorted(rpn_out.keys())

            # Aggregate RPN outputs to form proposals.
            rois = torch.cat(
                [generate_rois(rpn_out[k][1], layer_to_shifted_anchors[k], img_shape)
                 for k in sorted_rpn_keys], dim=0
            )
            scores = torch.cat(
                [get_scores(rpn_out[k][0]) for k in sorted_rpn_keys], dim=0
            )

            # For each image in the batch, apply NMS and select top-k proposals.
            unique_batches = rois[:, 0].unique()
            proposals_list = []
            for batch_idx in unique_batches:
                batch_mask = rois[:, 0] == batch_idx
                batch_rois = rois[batch_mask]
                batch_scores = scores[batch_mask]
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
                topk_proposals = sorted_proposals[:K]
                proposals_list.append(topk_proposals)

            if len(proposals_list) > 0:
                all_proposals = torch.cat(proposals_list, dim=0).detach()
            else:
                continue
            
            # print('='*10)
            # print(len(all_proposals))
            # print('='*10)

            # For each image in the batch, run the detection head to obtain final predictions.
            levels = sorted([int(x[-1]) for x in rpn_out.keys()])
            for batch_idx in unique_batches:
                img_idx = i * image_batch.shape[0] + int(batch_idx.item())
                image_id = image_id_map[img_idx]

                # Filter proposals for this image
                batch_mask = all_proposals[:, 0] == batch_idx
                batch_proposals = all_proposals[batch_mask]
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

                fg_mask = (labels > 0)
                final_boxes = decoded_boxes[fg_mask]
                final_scores = confs[fg_mask]
                final_labels = labels[fg_mask]

                if final_boxes.numel() == 0:
                    continue

                final_keep = ops.batched_nms(final_boxes, final_scores, final_labels, iou_threshold=0.5)
                final_boxes = final_boxes[final_keep]
                final_scores = final_scores[final_keep]
                final_labels = final_labels[final_keep]

                for box, score, label in zip(final_boxes, final_scores, final_labels):
                    # skip background or invalid label=0
                    label_int = int(label.item())
                    score_float = float(score.item())
                    x1, y1, x2, y2 = box.tolist()
                    width = x2 - x1
                    height = y2 - y1

                    if label_int == 0:
                        continue

                    coco_evaluator.predictions.append({
                        "image_id": image_id,
                        "category_id": label_int,  # or map label->category ID
                        "bbox": [x1, y1, width, height],
                        "score": score_float
                    })

    # After looping through the dataloader, create the COCO ground truth.
    # with open('temp_gt.json', 'w') as f:
    #     json.dump({
    #         "images": images,
    #         "annotations": annotations,
    #         "categories": categories
    #     }, f)
    # with open('temp_preds.json', 'w') as f:
    #     json.dump(coco_evaluator.predictions, f)
    with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as f:
        json.dump({
            "images": images,
            "annotations": annotations,
            "categories": categories
        }, f)
        temp_file_name = f.name

    # coco_gt = COCO('temp_gt.json')
    coco_gt = COCO(temp_file_name)
    coco_evaluator.coco_gt = coco_gt

    # Compute AP metrics for each specified IoU threshold.
    ap_metrics = coco_evaluator.compute_AP_for_thresholds(ap_iou_thresholds)
    ap_metrics_str = "\n".join(
        [f"{key:<20}: {value:.4f}" for key, value in sorted(ap_metrics.items())]
    )
    val_logger.info(f"COCO Evaluation AP Metrics for Epoch {epoch_num}:\n" + ap_metrics_str)

    # Clean up the temporary file.
    try:
        os.unlink(temp_file_name)
    except Exception:
        pass

    return ap_metrics