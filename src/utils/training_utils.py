import torch
import torch.nn.functional as F

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

def matching_and_sampling(all_proposals, all_scores, gt, num_samples):
    sampled_proposals_list = []
    batch_scores_list = []
    sampled_labels_list = []
    sampled_bbox_targets_list = []
    sampled_indices_list = []

    batch_ids = all_proposals[:, 0].unique()

    pos_iou_thresh = 0.5
    neg_iou_thresh = 0.1
    pos_fraction = 0.25

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

        # print(batch_proposals.size(), gt_b.size())
        
        # max_ious = torch.zeros(N_b, device=all_proposals.device)
        # gt_assignment = torch.full((N_b,), -1, dtype=torch.long, device=all_proposals.device)
        # for i in range(N_b):
        #     ious = compute_iou(batch_proposals[i], gt_b)  # shape [M_b]
        #     max_iou, idx = ious.max(0)
        #     max_ious[i] = max_iou
        #     gt_assignment[i] = idx
        ious = compute_iou_vectorized(batch_proposals, gt_b)  # [N_b, M_b]
        print(ious.size())
        max_ious, gt_assignment = ious.max(dim=1)  # max_ious: [N_b], gt_assignment: [N_b]

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
    sampled_proposals = torch.cat(sampled_proposals_list, dim=0)
    sampled_scores = torch.cat(batch_scores_list, dim=0)
    sampled_labels = torch.cat(sampled_labels_list, dim=0)
    sampled_bbox_targets = torch.cat(sampled_bbox_targets_list, dim=0)
    sampled_indices = torch.cat(sampled_indices_list, dim=0)

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
    cls_loss = F.cross_entropy(sampled_scores, sampled_labels)
    pos_inds = (sampled_labels == 1).nonzero(as_tuple=True)[0]

    pos_inds = torch.nonzero(sampled_labels > 0).squeeze(1)

    if pos_inds.numel() > 0:
        sampled_bbox_preds = sampled_bbox_preds.view(sampled_bbox_preds.size(0), num_classes, 4)
        pos_labels = sampled_labels[pos_inds].view(-1, 1, 1).expand(-1, 1, 4)
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