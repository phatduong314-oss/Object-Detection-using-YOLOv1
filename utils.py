import torch
import torch.nn as nn
from model_pretrained import YOLOV1
from metrics import IOU

def NMS(bboxes, iou_threshold, threshold, box_format="midpoint"):
    bboxes = [box for box in bboxes if box[1] > threshold]
    bboxes = sorted(bboxes, key=lambda x: x[1], reverse=True)
    bboxes_after_nms = []

    while bboxes:
        chosen_box = bboxes.pop(0)

        bboxes = [
            box
            for box in bboxes
            if box[0] != chosen_box[0]
            or IOU(
                torch.tensor(chosen_box[2:]).unsqueeze(0),
                torch.tensor(box[2:]).unsqueeze(0)
            ) < iou_threshold
        ]

        bboxes_after_nms.append(chosen_box)

    return bboxes_after_nms

def convert_cellboxes(predictions, S=7, C=20):
    """
    Chuyển đổi bounding boxes từ định dạng lưới (grid) về định dạng tỷ lệ so với toàn bộ ảnh (0 -> 1)
    """
    predictions = predictions.to("cpu")
    batch_size = predictions.shape[0]
    
    if predictions.shape[-1] == C + 5:  
        predictions = predictions.reshape(batch_size, S, S, C + 5)
        best_boxes = predictions[..., 21:25]
        scores = predictions[..., 20].unsqueeze(-1)
        classes = predictions[..., :C].argmax(-1).unsqueeze(-1).float()
    else:  # Predictions (30 chiều: 20 lớp + 2 x (1 obj + 4 bbox))
        predictions = predictions.reshape(batch_size, S, S, C + 10)
        bboxes1 = predictions[..., 21:25]
        bboxes2 = predictions[..., 26:30]
        
        scores = torch.cat(
            (predictions[..., 20].unsqueeze(0), predictions[..., 25].unsqueeze(0)), dim=0
        )
        best_box = scores.argmax(0).unsqueeze(-1)
        best_boxes = bboxes1 * (1 - best_box) + best_box * bboxes2
        
        scores = torch.max(predictions[..., 20], predictions[..., 25]).unsqueeze(-1)
        
        class_probs, classes = predictions[..., :C].max(-1)
        classes = classes.unsqueeze(-1).float()
        
        class_probs = class_probs.clamp(0.0, 1.0).unsqueeze(-1)
        scores = scores.clamp(0.0, 1.0) * class_probs
    
    cell_indices = torch.arange(S).repeat(batch_size, S, 1).unsqueeze(-1)
    
    x = 1 / S * (best_boxes[..., 0:1] + cell_indices)
    y = 1 / S * (best_boxes[..., 1:2] + cell_indices.permute(0, 2, 1, 3))
    w_y = best_boxes[..., 2:4]  
    
    converted_bboxes = torch.cat((classes, scores, x, y, w_y), dim=-1)
    
    return converted_bboxes

def cellboxes_to_boxes(out, S=7):
    converted_pred = convert_cellboxes(out).reshape(out.shape[0], S * S, -1)
    converted_pred[..., 0] = converted_pred[..., 0].long()
    all_bboxes = []

    for ex_idx in range(out.shape[0]):
        bboxes = []
        for bbox_idx in range(S * S):
            # bboxes.append([class_pred, prob_score, x, y, w, h])
            bboxes.append([x.item() for x in converted_pred[ex_idx, bbox_idx, :]])
        all_bboxes.append(bboxes)

    return all_bboxes

def get_bboxes(loader, model, iou_threshold, threshold, device="cuda", S=7, box_format="midpoint"):
    all_pred_boxes = []
    all_true_boxes = []

    model.eval()
    train_idx = 0

    with torch.no_grad():
        for batch_idx, (x, labels) in enumerate(loader):
            x = x.to(device)
            labels = labels.to(device)
            predictions = model(x)

            batch_size = x.shape[0]
            true_bboxes = cellboxes_to_boxes(labels)
            bboxes = cellboxes_to_boxes(predictions)

            for idx in range(batch_size):
                nms_boxes = NMS(
                    bboxes[idx],
                    iou_threshold=iou_threshold,
                    threshold=threshold,
                    box_format=box_format,
                )

                for nms_box in nms_boxes:
                    all_pred_boxes.append([train_idx] + nms_box)

                for box in true_bboxes[idx]:
                    if box[1] > threshold:  
                        all_true_boxes.append([train_idx] + box)

                train_idx += 1

    model.train()
    return all_pred_boxes, all_true_boxes



    





    

