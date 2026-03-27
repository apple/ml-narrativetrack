#
# For licensing see accompanying LICENSE file.
# Copyright (C) 2026 Apple Inc. All Rights Reserved.
#

import os
import json
import numpy as np
from collections import defaultdict
from statsmodels.stats.inter_rater import fleiss_kappa


def compute_fleiss_kappa(df):
    label_map = {'keep': 0, 'delete': 1, 'unsure': 2}
    df_num = df.applymap(lambda x: label_map.get(x, -1))
    num_items = df.shape[0]
    num_categories = 3
    fleiss_matrix = np.zeros((num_items, num_categories), dtype=int)

    for i, row in enumerate(df_num.itertuples(index=False)):
        for label in row:
            if label >= 0:
                fleiss_matrix[i][label] += 1
    return fleiss_kappa(fleiss_matrix)

def percentage_agreement(df):
    return (df.nunique(axis=1) == 1).mean()

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / (areaA + areaB - interArea + 1e-6)

def compute_set_miou(boxes_a, boxes_b):
    """
    Computes Set mIoU between two sets of boxes.
    boxes_a and boxes_b are lists of [x1, y1, x2, y2]
    Returns average IoU over best matched pairs (greedy matching)
    """
    if boxes_a.size(0) == 0 or boxes_b.size(0) == 0:
        return 0

    matched_iou = []
    used_b = set()

    for box_a in boxes_a:
        best_iou = 0
        best_j = -1
        for j, box_b in enumerate(boxes_b):
            if j in used_b:
                continue
            iou = compute_iou(box_a, box_b)
            if iou > best_iou:
                best_iou = iou
                best_j = j
        if best_j >= 0:
            used_b.add(best_j)
            matched_iou.append(best_iou)

    return sum(matched_iou) / max(len(boxes_a), len(boxes_b))

def compute_overall_metrics(results_by_timestamp):
    total_tp = 0
    total_gt = 0
    total_pred = 0
    total_miou = 0
    miou_count = 0

    for ts_result in results_by_timestamp.values():
        tp = ts_result["correct_detections"]
        gt = ts_result["total_gt"]
        pred = ts_result["total_pred"]
        miou = ts_result["mIoU"]

        total_tp += tp
        total_gt += gt
        total_pred += pred
        total_miou += miou
        if pred > 0 and tp > 0:
            miou_count += 1  # only count mIoU when there was a valid prediction

    overall_recall = total_tp / total_gt if total_gt else 0.0
    overall_precision = total_tp / total_pred if total_pred else 0.0
    average_miou = total_miou / miou_count if miou_count else 0.0

    return {
        "overall_correct_detections": total_tp,
        "overall_total_gt": total_gt,
        "overall_total_pred": total_pred,
        "overall_recall": overall_recall,
        "overall_precision": overall_precision,
        "average_mIoU": average_miou
    }

def compute_detection_metrics(gt_boxes, pred_boxes, iou_threshold=0.5):
    matched_gt = set()
    correct = 0
    ious = []

    for pred in pred_boxes:
        best_iou = 0
        best_gt_idx = -1
        for i, gt in enumerate(gt_boxes):
            if i in matched_gt:
                continue
            iou = compute_iou(pred, gt)
            
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = i

        if best_iou >= iou_threshold:
            correct += 1
            ious.append(best_iou)
            matched_gt.add(best_gt_idx)

    total_gt = len(gt_boxes)
    total_pred = len(pred_boxes)
    miou = np.mean(ious) if ious else 0.0

    return {
        "correct_detections": correct,
        "total_gt": total_gt,
        "total_pred": total_pred,
        "recall": max(correct / total_gt, 1.0) if total_gt else 0.0,
        "precision": correct / total_pred if total_pred else 0.0,
        "mIoU": miou
    }
    
def evaluate_entity_detection_per_timestamp(metadata, iou_threshold=0.5):
    results_by_timestamp = {}

    # Step 1: group all predicted and gt boxes by timestamp
    pred_by_ts = defaultdict(list)
    gt_by_ts = defaultdict(list)

    for name, entries in metadata.items():
        for entry in entries:
            ts = entry['timestamp']
            pred_by_ts[ts].append(entry['person_bbox'])
            gt_by_ts[ts].extend([g['bbox'] for g in entry['gt']])

    timestamps = sorted(set(gt_by_ts.keys()) | set(pred_by_ts.keys()))
    for ts in timestamps:
        pred_boxes = pred_by_ts.get(ts, [])
        gt_boxes = gt_by_ts.get(ts, [])
        results_by_timestamp[ts] = compute_detection_metrics(gt_boxes, pred_boxes, iou_threshold)

    overall = compute_overall_metrics(results_by_timestamp)
    return {
        **overall,
        "results_by_timestamp": results_by_timestamp,
    }

def evaluate_detection_per_timestamp(metadata, avg_set_miou, iou_threshold=0.5):
    results_by_timestamp = {}
    for entry in metadata:
        ts = entry['timestamp']
        pred_boxes = entry['pred_boxes']
        gt_boxes = [e['bbox'] for e in entry['gt_boxes']]
        results_by_timestamp[ts] = compute_detection_metrics(gt_boxes, pred_boxes, iou_threshold)

    overall = compute_overall_metrics(results_by_timestamp)
    return {
        **overall,
        "average_set_mIoU": avg_set_miou,
        "results_by_timestamp": results_by_timestamp,
    }

def evaluate_action_accuracy_per_entity(metadata, iou_threshold=0.5):
    action_by_timestamp = []
    debug_info = []

    for name, entries in metadata.items():
        for entry in entries:
            timestamp = entry['timestamp']
            pred_box = entry['person_bbox']
            pred_actions = [a['id'] for a in entry['actions']]
            gt_boxes = entry['gt']

            best_iou = 0
            best_gt_actions = []

            # Match with best IoU GT box
            for gt in gt_boxes:
                gt_box = gt['bbox']
                iou = compute_iou(pred_box, gt_box)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_actions = gt['action_ids'] if 'action_ids' in gt else [gt['label']]

            if best_iou >= iou_threshold:
                correct = sum(1 for p in pred_actions if p in best_gt_actions)
                acc = correct / len(pred_actions) if pred_actions else 0.0
                action_by_timestamp.append(acc)
                debug_info.append({
                    "timestamp": timestamp,
                    "name": name,
                    "gt": best_gt_actions,
                    "pred": pred_actions,
                    "correct": correct,
                    "accuracy": acc
                })

    average_accuracy = sum(action_by_timestamp) / len(action_by_timestamp) if action_by_timestamp else 0.0
    return {
        "average_topk_accuracy": average_accuracy,
        "n_entities": len(action_by_timestamp),
        "debug": debug_info  # optional: remove if large
    }

def calc_consecutive_duration(metadata):
    duration_stats = {}

    for entity, entries in metadata.items():
        timestamps = sorted(entry['timestamp'] for entry in entries)
        if not timestamps: continue

        segments = []
        current_segment = [timestamps[0]]

        for t in timestamps[1:]:
            if t == current_segment[-1] + 1:
                current_segment.append(t)
            else:
                segments.append(current_segment)
                current_segment = [t]

        if current_segment:
            segments.append(current_segment)

        # Compute durations
        durations = [seg[-1] - seg[0] + 1 for seg in segments]
        avg_duration = sum(durations) / len(durations)
        min_duration = min(durations)
        max_duration = max(durations)

        start = timestamps[0]
        end = timestamps[-1]
        lifetime_span = end - start + 1

        duration_stats[entity] = {
            "first_seen_timestamp": start,
            "last_seen_timestamp": end,
            "total_lifetime_span": lifetime_span,
            "avg_duration": avg_duration,
            "min_duration": min_duration,
            "max_duration": max_duration,
            "num_segments": len(segments),
        }

    return duration_stats

def avg_metrics(metadata_dir):
    recall, precision, miou = [], [], []
    f1 = []

    # Load metadata from all JSON files
    for subdir in os.listdir(metadata_dir):
        subdir_path = os.path.join(metadata_dir, subdir)
        if not os.path.isdir(subdir_path):
            continue

        meta_path = os.path.join(subdir_path, "detection_bbox_acc.json")
        if not os.path.isfile(meta_path):
            continue

        with open(meta_path, 'r') as f:
            data = json.load(f)
        
        recall.append(data['overall_recall'])
        precision.append(data['overall_precision'])
        miou.append(data['average_mIoU'])
        f1.append(2 * (data['overall_recall'] * data['overall_precision']) / (data['overall_recall'] + data['overall_precision']) if (data['overall_recall'] + data['overall_precision']) > 0 else 0)

    avg_recall = np.mean(recall) if recall else 0.0
    avg_precision = np.mean(precision) if precision else 0.0
    avg_miou = np.mean(miou) if miou else 0.0
    avg_f1 = np.mean(f1) if f1 else 0.0 
    print(f"Average Recall: {avg_recall:.4f}")
    print(f"Average Precision: {avg_precision:.4f}")    
    print(f"Average mIoU: {avg_miou:.4f}")
    print(f"Average F1 Score: {avg_f1:.4f}")
    