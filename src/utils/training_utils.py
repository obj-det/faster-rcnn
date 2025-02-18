import torch

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

def matching_and_sampling(all_proposals, gt):
    sampled_proposals_list = []
    sampled_labels_list = []
    sampled_bbox_targets_list = []

    batch_ids = all_proposals[:, 0].unique()

    pos_iou_thresh = 0.5
    neg_iou_thresh = 0.1
    pos_fraction = 0.25
    num_samples = 128

    for batch_idx in batch_ids:
        batch_mask = all_proposals[:, 0] == batch_idx
        batch_proposals = all_proposals[batch_mask][:, 1:]

        gt_mask = gt[:, 0] == batch_idx
        gt_b = gt[gt_mask][:, 1:5]
        gt_labels = gt[gt_mask][:, 5].long()

        N_b = batch_proposals.shape[0]
        M_b = gt_b.shape[0]

        # print(batch_proposals.size(), gt_b.size())
        
        max_ious = torch.zeros(N_b, device=all_proposals.device)
        gt_assignment = torch.full((N_b,), -1, dtype=torch.long, device=all_proposals.device)
        for i in range(N_b):
            ious = compute_iou(batch_proposals[i], gt_b)  # shape [M_b]
            max_iou, idx = ious.max(0)
            max_ious[i] = max_iou
            gt_assignment[i] = idx

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
        
        # print(bbox_targets_b.size(), labels_b.size())
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

        proposals_keep = batch_proposals[keep_inds]  # shape [K_b, 4]
        labels_keep = labels_b[keep_inds]         # shape [K_b]
        bbox_targets_keep = bbox_targets_b[keep_inds]  # shape [K_b, 4]
        
        batch_idx_tensor = torch.full((proposals_keep.shape[0], 1), batch_idx, device=all_proposals.device)
        proposals_keep = torch.cat([batch_idx_tensor, proposals_keep], dim=1)  # now [K_b, 5]

        sampled_proposals_list.append(proposals_keep)
        sampled_labels_list.append(labels_keep)
        sampled_bbox_targets_list.append(bbox_targets_keep)

    # Concatenate results from all batches.
    sampled_proposals = torch.cat(sampled_proposals_list, dim=0)
    sampled_labels = torch.cat(sampled_labels_list, dim=0)
    sampled_bbox_targets = torch.cat(sampled_bbox_targets_list, dim=0)

    return sampled_proposals, sampled_labels, sampled_bbox_targets