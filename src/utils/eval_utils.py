"""
Evaluation utilities for Faster R-CNN object detection.

This module provides utilities for evaluating object detection models using COCO metrics:
- Average Precision (AP) calculation
- Support for multiple IoU thresholds
- Evaluation across different object sizes (small, medium, large)
- Batch prediction accumulation
"""

import torch
from src.utils.anchor_utils import *

import numpy as np
from pycocotools.cocoeval import COCOeval

class CocoEvaluator:
    """COCO-style evaluator for object detection models.
    
    This class handles:
    - Accumulating predictions across batches
    - Converting predictions to COCO format
    - Computing AP metrics using pycocotools
    - Supporting evaluation at different IoU thresholds
    - Breaking down performance by object size
    """
    
    def __init__(self, coco_gt):
        """Initialize the COCO evaluator.
        
        Args:
            coco_gt (COCO): COCO ground truth object containing the full evaluation annotations.
                           This should be created using the pycocotools.coco.COCO class.
        """
        self.coco_gt = coco_gt
        self.predictions = []

    def update(self, proposals, bbox_deltas, cls_scores, image_id, im_shape):
        """Store predictions for evaluation after decoding bbox deltas.
        
        This method:
        1. Decodes final boxes using proposals and bbox deltas
        2. Computes class scores and selects highest probability class
        3. Converts boxes to COCO format (x, y, width, height)
        4. Stores predictions for later evaluation
        
        Args:
            proposals (torch.Tensor): Proposals of shape [N, 5] containing
                                    (batch_idx, x1, y1, x2, y2)
            bbox_deltas (torch.Tensor): Predicted bbox deltas of shape [N, 4]
            cls_scores (torch.Tensor): Classification scores for each proposal
            image_id (int): COCO image ID for this set of predictions
            im_shape (tuple): Image dimensions (height, width)
        """
        # Decode final boxes using the proposals (excluding batch index) and bbox deltas
        decoded_boxes = decode_boxes(proposals[:, 1:], bbox_deltas, im_shape)
        
        # Get highest probability class for each proposal
        scores, labels = torch.max(torch.softmax(cls_scores, dim=1), dim=1)
        
        # Convert to numpy for COCO format
        boxes = decoded_boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # Store predictions in COCO format
        for box, score, label in zip(boxes, scores, labels):
            if int(label) == 0:  # Skip background predictions
                continue

            self.predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0]),
                    float(box[3] - box[1])
                ],
                "score": float(score)
            })

    def compute_AP(self, iou_thresh=None):
        """Compute COCO AP metrics.
        
        Calculates Average Precision metrics using COCO evaluation protocol.
        Can evaluate at a specific IoU threshold or use the default COCO thresholds.
        
        Args:
            iou_thresh (float, optional): Specific IoU threshold for evaluation.
                                        If None, uses default COCO thresholds.
        
        Returns:
            dict: Dictionary containing AP metrics:
                If iou_thresh is None:
                    - AP_small: AP for small objects
                    - AP_medium: AP for medium objects
                    - AP_large: AP for large objects
                If iou_thresh is specified:
                    - AP: AP at the specified IoU threshold
                    - AP_small: AP for small objects
                    - AP_medium: AP for medium objects
                    - AP_large: AP for large objects
        """
        # Handle empty predictions
        if len(self.predictions) == 0:
            if iou_thresh is None:
                return {"AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}
            else:
                return {"AP": 0.0, "AP_small": 0.0, "AP_medium": 0.0, "AP_large": 0.0}

        # Load predictions and create evaluator
        coco_dt = self.coco_gt.loadRes(self.predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        
        # Set IoU threshold if specified
        if iou_thresh is not None:
            coco_eval.params.iouThrs = np.array([iou_thresh])
        
        # Run evaluation
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Extract metrics based on evaluation mode
        if iou_thresh is None:
            # Using default COCO thresholds
            ap_small = coco_eval.stats[3] if len(coco_eval.stats) > 3 else None
            ap_medium = coco_eval.stats[4] if len(coco_eval.stats) > 4 else None
            ap_large = coco_eval.stats[5] if len(coco_eval.stats) > 5 else None
            return {"AP_small": ap_small, "AP_medium": ap_medium, "AP_large": ap_large}
        else:
            # Using single IoU threshold
            ap = coco_eval.stats[0] if len(coco_eval.stats) > 0 else None
            ap_small = coco_eval.stats[3] if len(coco_eval.stats) > 3 else None
            ap_medium = coco_eval.stats[4] if len(coco_eval.stats) > 4 else None
            ap_large = coco_eval.stats[5] if len(coco_eval.stats) > 5 else None
            return {"AP": ap, "AP_small": ap_small, "AP_medium": ap_medium, "AP_large": ap_large}

    def compute_AP_for_thresholds(self, iou_thresholds):
        """Compute COCO AP metrics for multiple IoU thresholds.
        
        Args:
            iou_thresholds (list): List of IoU thresholds to evaluate at
        
        Returns:
            dict: Dictionary containing AP metrics for each threshold.
                 Keys are formatted as:
                 - AP_{threshold}
                 - AP_small_{threshold}
                 - AP_medium_{threshold}
                 - AP_large_{threshold}
        """
        results = {}
        for thr in iou_thresholds:
            metrics = self.compute_AP(iou_thresh=thr)
            results[f"AP_{thr:.2f}"] = metrics["AP"]
            results[f"AP_small_{thr:.2f}"] = metrics["AP_small"]
            results[f"AP_medium_{thr:.2f}"] = metrics["AP_medium"]
            results[f"AP_large_{thr:.2f}"] = metrics["AP_large"]
        return results

    def reset(self):
        """Clear accumulated predictions.
        
        Call this method between evaluation runs to reset the evaluator state.
        """
        self.predictions.clear()