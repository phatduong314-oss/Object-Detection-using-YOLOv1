import torch
import torch.nn as nn
from collections import Counter

#Hyperparameter from the original paper
S = 7
B = 2
C = 20
lambda_coord = 5 
lambda_noobj = 0.5

def IOU(boxes_preds, boxes_labels, box_format="midpoint"):
    if box_format == "midpoint":
        box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
        box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
        box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
        box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2
        
        box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
        box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
        box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
        box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2
        
    elif box_format == "corners":
        box1_x1 = boxes_preds[..., 0:1]
        box1_y1 = boxes_preds[..., 1:2]
        box1_x2 = boxes_preds[..., 2:3]
        box1_y2 = boxes_preds[..., 3:4]  
        box2_x1 = boxes_labels[..., 0:1]
        box2_y1 = boxes_labels[..., 1:2]
        box2_x2 = boxes_labels[..., 2:3]
        box2_y2 = boxes_labels[..., 3:4]

    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)
    box1_area = abs((box1_x2 - box1_x1) * (box1_y2 - box1_y1))
    box2_area = abs((box2_x2 - box2_x1) * (box2_y2 - box2_y1))
    union = box1_area + box2_area - intersection + 1e-6

    return intersection / union

def mAP(pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20):
    avg_precisions = []

    for c in range(num_classes):
        detections = []
        ground_truths = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for true_box in true_boxes:
            if true_box[1] == c:
                ground_truths.append(true_box)

        amount_bboxes = Counter([gt[0] for gt in ground_truths])

        for key, val in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(val)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros((len(detections)))
        FP = torch.zeros((len(detections)))
        total_true_bboxes = len(ground_truths)
        
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            ground_truth_img = [bbox for bbox in ground_truths if bbox[0] == detection[0]]
            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = IOU(
                    torch.tensor(detection[3:]),
                    torch.tensor(gt[3:]),
                    box_format=box_format,
                )

                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1
            else:
                FP[detection_idx] = 1

        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)
        recalls = TP_cumsum / (total_true_bboxes + 1e-6)
        precisions = torch.divide(TP_cumsum, (TP_cumsum + FP_cumsum + 1e-6))
        precisions = torch.cat((torch.tensor([1]), precisions))
        recalls = torch.cat((torch.tensor([0]), recalls))
        avg_precisions.append(torch.trapz(precisions, recalls))

    return sum(avg_precisions) / len(avg_precisions)

def loss(target, prediction):
    sse = nn.MSELoss(reduction='sum')
    target = target.reshape(-1, S, S, C + 5)
    prediction = prediction.reshape(-1, S, S, C + B * 5)
    batch_size = target.shape[0]

    obj = target[..., 20].unsqueeze(3)
    noobj = 1 - obj

    target_bbox = target[..., 21:25]
    pred_bbox0 = prediction[..., 21:25]
    pred_bbox1 = prediction[..., 26:30]

    # Convert to image-relative coords for proper IOU calculation
    cell_indices = torch.arange(S, device=target.device).repeat(batch_size, S, 1).unsqueeze(-1)
    
    tgt_x = 1/S * (target_bbox[..., 0:1] + cell_indices)
    tgt_y = 1/S * (target_bbox[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    converted_tgt = torch.cat([tgt_x, tgt_y, target_bbox[..., 2:4]], dim=-1)
    
    pb0_x = 1/S * (pred_bbox0[..., 0:1] + cell_indices)
    pb0_y = 1/S * (pred_bbox0[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    converted_pb0 = torch.cat([pb0_x, pb0_y, pred_bbox0[..., 2:4]], dim=-1)
    
    pb1_x = 1/S * (pred_bbox1[..., 0:1] + cell_indices)
    pb1_y = 1/S * (pred_bbox1[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    converted_pb1 = torch.cat([pb1_x, pb1_y, pred_bbox1[..., 2:4]], dim=-1)

    iou_pred_bbox0 = IOU(converted_pb0, converted_tgt)
    iou_pred_bbox1 = IOU(converted_pb1, converted_tgt)
    iou_pred_bboxes = torch.cat(
        [iou_pred_bbox0.unsqueeze(0), iou_pred_bbox1.unsqueeze(0)],
        dim=0,
    )

    best_iou, best_bbox_idx = torch.max(iou_pred_bboxes, dim=0)

    target_bbox = obj * target_bbox
    best_bbox = obj * (best_bbox_idx * pred_bbox1 + (1 - best_bbox_idx) * pred_bbox0)

    target_bbox = torch.cat(
        [
            target_bbox[..., 0:2],
            torch.sqrt(target_bbox[..., 2:4].clamp(min=1e-6)),
        ],
        dim=-1,
    )
    best_bbox = torch.cat(
        [
            best_bbox[..., 0:2],
            torch.sign(best_bbox[..., 2:4]) * torch.sqrt(torch.abs(best_bbox[..., 2:4]) + 1e-6),
        ],
        dim=-1,
    )

    bbox_loss = sse(
        torch.flatten(target_bbox, end_dim=-2),
        torch.flatten(best_bbox, end_dim=-2),
    )

    target_bbox_confidence = target[..., 20:21]
    pred_bbox0_confidence = prediction[..., 20:21]
    pred_bbox1_confidence = prediction[..., 25:26]

    target_bbox_confidence = obj * target_bbox_confidence
    best_bbox_confidence = obj * (
        best_bbox_idx * pred_bbox1_confidence + (1 - best_bbox_idx) * pred_bbox0_confidence
    )

    object_loss = sse(
        torch.flatten(obj * target_bbox_confidence * best_iou.detach()),
        torch.flatten(obj * best_bbox_confidence),
    )

    no_object_loss = sse(
        torch.flatten(noobj * target_bbox_confidence),
        torch.flatten(noobj * pred_bbox0_confidence),
    )
    no_object_loss += sse(
        torch.flatten(noobj * target_bbox_confidence),
        torch.flatten(noobj * pred_bbox1_confidence),
    )

    target_class = target[..., :20]
    pred_class = prediction[..., :20]

    class_loss = sse(
        torch.flatten(obj * target_class, end_dim=-2),
        torch.flatten(obj * pred_class, end_dim=-2),
    )

    total_loss = (
        lambda_coord * bbox_loss
        + object_loss
        + lambda_noobj * no_object_loss
        + class_loss
    ) / batch_size

    return bbox_loss, object_loss, no_object_loss, class_loss, total_loss
