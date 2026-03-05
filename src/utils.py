"""
utils.py — Utility functions

Contains:
- Evaluation metrics computation (IoU, Dice, Detection Accuracy, F1, Confusion Matrix)
- Bounding box IoU computation
- Other helper functions
"""

import numpy as np
import torch
import cv2


# ============================================================
# Segmentation metrics
# ============================================================
def compute_iou(pred_mask, gt_mask):
    """
    Compute IoU (Intersection over Union) for a single image.
    
    IoU measures the overlap between predicted and ground truth regions:
    IoU = |Pred ∩ GT| / |Pred ∪ GT|
    - IoU = 1.0: perfect match
    - IoU = 0.0: no overlap

    Args:
        pred_mask: numpy array (H, W), binary 0/1
        gt_mask:   numpy array (H, W), binary 0/1
    
    Returns:
        iou: float
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    intersection = (pred & gt).sum()
    union = (pred | gt).sum()
    
    if union == 0:
        return 1.0 if intersection == 0 else 0.0  # Both empty → IoU=1
    
    return intersection / union


def compute_dice(pred_mask, gt_mask):
    """
    Compute Dice coefficient.
    
    Dice = 2 * |Pred ∩ GT| / (|Pred| + |GT|)
    Similar to IoU but with higher values (Dice ≥ IoU), commonly used in medical image segmentation.

    Args:
        pred_mask, gt_mask: numpy array (H, W), binary 0/1
    
    Returns:
        dice: float
    """
    pred = pred_mask.astype(bool)
    gt = gt_mask.astype(bool)
    
    intersection = (pred & gt).sum()
    total = pred.sum() + gt.sum()
    
    if total == 0:
        return 1.0
    
    return 2.0 * intersection / total


# ============================================================
# Detection metrics
# ============================================================
def compute_bbox_iou(bbox1, bbox2):
    """
    Compute IoU between two bounding boxes.
    
    Bounding box format: (x1, y1, x2, y2)

    Args:
        bbox1, bbox2: tuple (x1, y1, x2, y2)
    
    Returns:
        iou: float
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    # Intersection area
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    # Individual areas
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    # Union area
    union = area1 + area2 - intersection
    
    if union == 0:
        return 0.0
    
    return intersection / union


def mask_to_bbox(mask):
    """
    Extract bounding box from binary mask.
    
    Args:
        mask: numpy array (H, W), 0/1 or 0/255

    Returns:
        (x1, y1, x2, y2) or None
    """
    if mask.dtype != np.uint8:
        mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    else:
        mask_uint8 = mask
    
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) == 0:
        return None
    
    all_points = np.concatenate(contours, axis=0)
    x, y, w, h = cv2.boundingRect(all_points)
    return (x, y, x + w, y + h)


def detection_accuracy(pred_bboxes, gt_bboxes, iou_threshold=0.5):
    """
    Compute Detection Accuracy @ IoU threshold.
    
    A detection is considered successful if the IoU between predicted and GT bbox >= threshold.

    Args:
        pred_bboxes: list of (x1,y1,x2,y2) or None
        gt_bboxes:   list of (x1,y1,x2,y2) or None
        iou_threshold: IoU threshold

    Returns:
        accuracy: float (0~1)
    """
    correct = 0
    total = len(gt_bboxes)
    
    for pred, gt in zip(pred_bboxes, gt_bboxes):
        if pred is None or gt is None:
            continue
        if compute_bbox_iou(pred, gt) >= iou_threshold:
            correct += 1
    
    return correct / total if total > 0 else 0.0


# ============================================================
# Classification metrics
# ============================================================
def compute_confusion_matrix(all_preds, all_labels, num_classes):
    """
    Compute confusion matrix.
    
    confusion_matrix[i][j] = number of samples with true class i predicted as class j.
    Diagonal values represent correctly classified samples.

    Args:
        all_preds:   list/array of predicted labels
        all_labels:  list/array of ground truth labels
        num_classes: total number of classes

    Returns:
        cm: numpy array (num_classes, num_classes)
    """
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for pred, gt in zip(all_preds, all_labels):
        cm[gt][pred] += 1
    return cm


def compute_f1_macro(confusion_matrix):
    """
    Compute Macro-averaged F1 Score from confusion matrix.
    
    Macro F1 = average of F1 scores computed for each class separately.
    F1 for each class = 2 * Precision * Recall / (Precision + Recall)
    
    This metric treats all classes equally and is not affected by class imbalance.

    Args:
        confusion_matrix: (num_classes, num_classes)
    
    Returns:
        macro_f1: float
        per_class_f1: list of float
    """
    num_classes = confusion_matrix.shape[0]
    per_class_f1 = []
    
    for c in range(num_classes):
        tp = confusion_matrix[c, c]  # True positives
        fp = confusion_matrix[:, c].sum() - tp  # False positives (other classes predicted as c)
        fn = confusion_matrix[c, :].sum() - tp  # False negatives (c predicted as other classes)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class_f1.append(f1)
    
    macro_f1 = np.mean(per_class_f1)
    return macro_f1, per_class_f1
