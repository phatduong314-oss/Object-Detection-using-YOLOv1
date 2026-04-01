import torch
import cv2
import torchvision.transforms as transforms
from model_pretrained import YOLOV1
from dataset import VOC_classes, IMAGE_SIZE, test_transform
from utils import NMS, convert_cellboxes

def predict(image_path, model_path="best_yolo_model.pth"):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOV1(pretrained=False).to(device)
    # Load model weights (changes restored)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    original_img = cv2.imread(image_path)        
    original_h, original_w, _ = original_img.shape
    rgb_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    resized_img = cv2.resize(rgb_img, (IMAGE_SIZE, IMAGE_SIZE))
    
    # Sử dụng transform của torchvision (như đã thiết lập ở dataset.py)
    input_tensor = test_transform(resized_img).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        out = model(input_tensor)

    bboxes = convert_cellboxes(out).reshape(-1, 6)
    
    bboxes_list = bboxes.tolist()
    
    max_prob = max([box[1] for box in bboxes_list]) if bboxes_list else 0
    print(f"DEBUG: Maximum confidence score predicted by model: {max_prob:.4f}")
    
    # Lower threshold to see if ANY boxes are being generated
    final_boxes = NMS(bboxes_list, iou_threshold=0.5, threshold=0.4)
    print(f"DEBUG: Number of boxes after NMS: {len(final_boxes)}")

    # Draw boxes
    for box in final_boxes:
        class_pred = int(box[0])
        prob_score = box[1]
        x, y, w, h = box[2], box[3], box[4], box[5]
        
        # Un-normalize back to original image size
        xmin = int((x - w / 2) * original_w)
        ymin = int((y - h / 2) * original_h)
        xmax = int((x + w / 2) * original_w)
        ymax = int((y + h / 2) * original_h)

        class_name = VOC_classes[class_pred]
        
        cv2.rectangle(original_img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(original_img, f"{class_name}: {prob_score:.2f}", (xmin, ymin - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # cv2.imshow("Detection", original_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    cv2.imwrite("output_detection.jpg", original_img)
    print("Prediction complete! Output saved to output_detection.jpg")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run YOLOv1 Inference")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="best_yolo_model.pth", help="Path to model weights")
    args = parser.parse_args()
    
    predict(args.image, args.weights)