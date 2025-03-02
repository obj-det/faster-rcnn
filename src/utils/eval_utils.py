import torch
import torch.nn.functional as F
from tqdm import tqdm
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO

from src.utils.anchor_utils import *

class CocoEvaluator:
    def __init__(self, coco_gt):
        """
        Args:
            coco_gt (COCO): COCO ground truth object containing the full evaluation annotations.
        """
        self.coco_gt = coco_gt
        self.predictions = []

    def update(self, proposals, bbox_deltas, cls_scores, gt, im_shape):
        """
        Store predictions for evaluation after decoding bbox deltas.

        Args:
            proposals (Tensor): [N, 5] tensor of proposals (batch_idx, x1, y1, x2, y2).
            bbox_deltas (Tensor): Predicted bbox deltas (shape: [N, 4]).
            cls_scores (Tensor): Predicted classification scores.
            gt (Tensor): Ground truth boxes for the image.
                       Assumes the first column contains the image_id.
            im_shape (tuple): (height, width) of the image.
        """
        # Decode final boxes using the proposals (excluding batch index) and bbox deltas.
        decoded_boxes = decode_boxes(proposals[:, 1:], bbox_deltas, im_shape)
        # For each proposal, select the class with the highest probability.
        scores, labels = torch.max(torch.softmax(cls_scores, dim=1), dim=1)
        boxes = decoded_boxes.detach().cpu().numpy()
        scores = scores.detach().cpu().numpy()
        labels = labels.detach().cpu().numpy()

        # NOTE: This assumes a single image per batch.
        image_id = int(gt[0, 0].item())
        for box, score, label in zip(boxes, scores, labels):
            self.predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [
                    float(box[0]),
                    float(box[1]),
                    float(box[2] - box[0] + 1),
                    float(box[3] - box[1] + 1)
                ],
                "score": float(score)
            })

    def compute_AP(self):
        """Compute COCO AP metrics and return AP for small, medium, and large objects."""
        coco_dt = self.coco_gt.loadRes(self.predictions)
        coco_eval = COCOeval(self.coco_gt, coco_dt, "bbox")
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()

        # Indices: 3 for small, 4 for medium, 5 for large objects.
        ap_small = coco_eval.stats[3]
        ap_medium = coco_eval.stats[4]
        ap_large = coco_eval.stats[5]

        return {"AP_small": ap_small, "AP_medium": ap_medium, "AP_large": ap_large}

    def reset(self):
        """Clear predictions."""
        self.predictions.clear()
