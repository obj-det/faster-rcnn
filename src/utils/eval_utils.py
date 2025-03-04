import torch
from src.utils.anchor_utils import *

import numpy as np
from pycocotools.cocoeval import COCOeval

class CocoEvaluator:
    def __init__(self, coco_gt):
        """
        Args:
            coco_gt (COCO): COCO ground truth object containing the full evaluation annotations.
        """
        self.coco_gt = coco_gt
        self.predictions = []

    def update(self, proposals, bbox_deltas, cls_scores, image_id, im_shape):
        """
        Store predictions for evaluation after decoding bbox deltas.

        Args:
            proposals (Tensor): [N, 5] tensor of proposals (batch_idx, x1, y1, x2, y2).
            bbox_deltas (Tensor): Predicted bbox deltas (shape: [N, 4]).
            cls_scores (Tensor): Predicted classification scores.
            image_id (int): The COCO image ID for this set of predictions.
            im_shape (tuple): (height, width) of the image.
        """
        # Decode final boxes using the proposals (excluding batch index) and bbox deltas.
        decoded_boxes = decode_boxes(proposals[:, 1:], bbox_deltas, im_shape)
        # For each proposal, select the class with the highest probability.
        scores, labels = torch.max(torch.softmax(cls_scores, dim=1), dim=1)
        boxes = decoded_boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        pos_scores = scores[labels == 1]

        # print('='*50)
        # print(pos_scores.min(), pos_scores.max(), pos_scores.mean(), pos_scores.std())
        # quantiles = np.quantile(pos_scores, [0.25, 0.5, 0.75])
        # print("25th percentile:", quantiles[0])
        # print("Median:", quantiles[1])
        # print("75th percentile:", quantiles[2])
        # print(len(labels == 1))
        # print('='*50)

        # Add predictions using the provided image_id
        for box, score, label in zip(boxes, scores, labels):
            if int(label) == 0:
                # Skip predictions where the model predicts background.
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
        """
        Compute COCO AP metrics. If iou_thresh is provided, override the default IoU thresholds.

        Returns:
            dict: Dictionary containing AP metrics.
                  If iou_thresh is None, returns metrics averaged over the default range.
                  Otherwise returns metrics for that specific IoU threshold.
        """

        coco_dt = self.coco_gt.loadRes(self.predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        if iou_thresh is not None:
            coco_eval.params.iouThrs = np.array([iou_thresh])
        coco_eval.evaluate()
        coco_eval.accumulate()
        # Optionally, you can call coco_eval.summarize() to print the summary.
        coco_eval.summarize()

        if iou_thresh is None:
            # Use default summary indices (using average over multiple thresholds)
            ap_small = coco_eval.stats[3] if len(coco_eval.stats) > 3 else None
            ap_medium = coco_eval.stats[4] if len(coco_eval.stats) > 4 else None
            ap_large = coco_eval.stats[5] if len(coco_eval.stats) > 5 else None
            return {"AP_small": ap_small, "AP_medium": ap_medium, "AP_large": ap_large}
        else:
            # When using a single IoU threshold, stats[0] is the AP at that threshold.
            # The indices for small, medium, large might still be in positions 3, 4, 5.
            ap = coco_eval.stats[0] if len(coco_eval.stats) > 0 else None
            ap_small = coco_eval.stats[3] if len(coco_eval.stats) > 3 else None
            ap_medium = coco_eval.stats[4] if len(coco_eval.stats) > 4 else None
            ap_large = coco_eval.stats[5] if len(coco_eval.stats) > 5 else None
            return {"AP": ap, "AP_small": ap_small, "AP_medium": ap_medium, "AP_large": ap_large}

    def compute_AP_for_thresholds(self, iou_thresholds):
        """
        Compute COCO AP metrics for each IoU threshold provided.

        Args:
            iou_thresholds (list): List of IoU thresholds.

        Returns:
            dict: A dictionary with keys for each IoU threshold containing AP metrics.
                  For example: {"AP_0.50": 0.123, "AP_small_0.50": 0.045, ...}
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
        """Clear predictions."""
        self.predictions.clear()