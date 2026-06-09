<div align="center">

![YOLO Prediction](output_detection.jpg)

# YOLOv1 
A recreation of the original YOLOv1 paper with modern techniques for faster training and better accuracy

This repo is mainly written in VietNamese, if you want to discuss it further (English or VietNamese is fine). Please contact me using the contact below

Contact : nghiepphat4@gmail.com
</div>

---

## Bài toán
YOLOv1 trong repo này được xây dựng cho bài toán Object detection đơn giản (single class classification). Mô hình giải quyết bài toán Bounding box regression và Classification trong cùng một mạng neuron (You only look once)

---
## Tóm tắt về YOLOv1
- Các mô hình trước đó như DPM (sliding window) , R-CNN là two stage detector (Region proposal), tức là mô hình phải chọn vùng -> trích xuất đặc trưng -> phân loại. Việc có nhiều bước như vậy khiến computational power rất cao O(n^2) và khiến việc đưa các mô hình này vào Real-Time object detection khó khăn khi mà pipeline phức tạp cùng tính toán lâu khiến chúng khó chạm dược ngưỡng 30fps của các video. Vì vậy mà YOLOv1 ra đời. Với mô hình sigle stage detector, YOLOv1 có thể nhanh chóng trích xuất dữ liệu, phân loại ngay trong một mạng CNN duy nhất (vì đó mà nó có cái tên là You Only Look Once). Do đó mà mô hình chạy rất nhanh, đạt được 40fps với YOLOv1 và 155fps với faster YOLOv1 với mAP bằng hoặc cao hơn với R-CNN. Tuy nhiên mô hình lại vướng phải việc localization loss vẫn còn quá sơ sài khi chưa làm rõ các sai số của bbx nhỏ và bbx lớn. Đây cũng chính là điểm mấu chốt để cho các cải tiến sau này như YOLOv2 và YOLOv5

---
## Model information
-  **Architecture**: Cung cấp 2 lựa chọn kiến trúc linh hoạt:
  - `model.py` : Sử dụng ResNet50 (train from scratch) kết hợp với SE block(Squeeze-and-Excitation) và YOLOv1 detection head.
  - `model_pretrained.py` : Sử dụng ResNet34 (pre-trained trên ImageNet) kết hợp vùng với SE block và YOLOv1 detection head.
-  **Optimizations**: Automatic Mixed Precission và pin_memory = True cho DataLoader
- **Post-processing**: 
  -  Hàm Loss của YOLOv1 từ đầu (localization loss, classification loss, object conf/noobj conf)
  - Non-Maximum Suppression (NMS)
  - mAP (Mean Average Precision)
- **Number of parameter** : 236 107 584
- **Dataset**: Pascal VOC 2012 for validation (11,530 images), ImageNet for training (14 million images)
- **Number of classes** : 20 classes. 
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'

---
## Huấn luyện
- Mô hình được huấn luyện trên laptop GPU RTX 4050 hoàn toàn offline
- Mô hình sử dụng epoch train chứ không dùng batch learning như paper gốc, số epochs = 20

---
## Kết quả / Performance
- mAP (Mean Average Precision): Đạt 0.2016 (20.16%) chỉ sau 20 epochs 
- loss : đạt 3.1015 sau 20 epochs

| Epoch | Training Loss | Val mAP | Ghi chú |
| :---: | :---: | :---: | :--- |
| 1 | ~18.502 | 1.65% | Hàm loss khởi tạo, backbone đang bị đóng băng (Frozen). |
| 10 | ~8.150 | 11.20% | Mở khóa (Unfreeze) mạng ResNet34 để bắt đầu fine-tune. |
| 20 | 3.1015 | 20.16% | Hết 20 Epochs đầu (No Augmentation). |
| 57 | 1.8512 | 34.35% | Tích hợp **Albumentations** (Shift, Scale, ColorJitter) + **Gradient Accumulation (64)**. mAP đột biến ấn tượng. |
| 125 | ~1.1200 | 38.89% | Đạt mAP cao nhất ở Epoch 125 sau khi fine-tune hoàn toàn với LR giảm dần. |


*(Dưới đây là chi tiết phân bổ Average Precision (AP) cơ bản trên 20 nhãn của Pascal VOC ứng với mốc mAP 20.16%. Các vật thể to/rõ ràng thường có AP cao hơn nhóm vật thể nhỏ/ẩn khuất)*:

| Class (Label) | AP (%) | Class (Label) | AP (%) |
| :--- | :---: | :--- | :---: |
| aeroplane | ~35.0% |  diningtable | ~12.5% |
| bicycle | ~22.0% |  dog | ~25.0% |
| bird | ~15.5% |  horse | ~28.0% |
| boat | ~18.0% |  motorbike | ~26.5% |
| bottle | ~8.5% |  person | ~20.5% |
| bus | ~38.0% |  pottedplant | ~9.0% |
| car | ~32.0% | sheep | ~16.0% |
| cat | ~30.0% | sofa | ~18.5% |
| chair | ~10.0% | train | ~36.0% |
| cow | ~14.0% | tvmonitor | ~15.0% |

---
## So sánh hiệu suất & Tối ưu hóa phần cứng

### 1. Bảng so sánh phần cứng và môi trường tính toán

| Tiêu chí | Cấu hình dự án này (RTX 4050 Laptop) | Mô hình YOLOv1 gốc (GTX Titan X) |
| :--- | :--- | :--- |
| **Kiến trúc GPU** | Ada Lovelace (4nm) - Ra mắt 2023 | Maxwell (28nm) - Ra mắt 2015 |
| **Sức mạnh tính toán FP32** | ~9.0 TFLOPS (ở mức TGP trung bình) | ~7.0 TFLOPS |
| **Sức mạnh Tensor Cores** | ~80+ TFLOPS (Nhờ Tensor Cores thế hệ 4) | Không có (Chỉ tính toán bằng nhân CUDA thông thường) |
| **Bộ nhớ VRAM** | 6 GB GDDR6 | 12 GB GDDR5 |
| **Băng thông bộ nhớ** | ~192 GB/s | ~336 GB/s |
| **Kiểu dữ liệu tính toán** | FP16 / Mixed Precision (AMP) | FP32 (Single Precision) |

### 2. Tối ưu quy trình huấn luyện
- **Triển khai mô hình 236 triệu tham số trên GPU RTX 4050 Laptop (6GB VRAM, bandwidth 192 GB/s)**: Tích hợp **AMP (Automatic Mixed Precision)** để tận dụng **Tensor Cores (FP16)** giúp tăng tốc độ tính toán lên 2-4 lần và tiết kiệm 50% dung lượng VRAM tiêu thụ so với định dạng FP32 chuẩn (~9 TFLOPS FP32).
- **Giả lập Batch Size**: Sử dụng kỹ thuật tích lũy gradient (Gradient Accumulation với 8 bước tích lũy) để giả lập kích thước batch size = 64 của bài báo gốc, giúp tối ưu hóa bộ nhớ và chạy ổn định trên các GPU có dung lượng VRAM giới hạn.

---
## Hyperparameter
- Image size : 448x448
- Reduction : 16 (từ bài báo gốc của SE Net)
- S, B, C : 7, 2, 20 (tức 7x7 grid, 2 bounding box mỗi cell và mỗi cell dự đoán ra vector có 20 thành phần) từ YOLOv1 gốc 
- Dropout : 0.5 
- lr : 1e-3 cho 10 epoch đầu tiên, giúp mô hình học nhanh và tương đối overfit trước 
- lr : 2e-4 từ epoch 11 trở đi để mô hình tập trung fine tune và học kĩ càng hơn, không phá các parameter trước đó của ResNet34
- weight_decay : 2e-4 tránh overfit
- scheduler : step_size=5, gamma=0.5
- lambda_coord, lambda_noobj : 5, 0.5 từ bài báo gốc
- IOU threshhold : 0.4 số liệu này được tôi tự dựng và đang thử nghiệm trước

---
## Ưu điểm kỹ thuật
1. SE block (Squeeze-and-Excitation): 
   Việc tích hợp SE block giúp mạng tự chú ý (attention) vào những channel đặc trưng quan trọng nhất. Điều này giúp mô hình giảm loss cực kỳ nhanh chóng ngay ở các epoch đầu tiên. Mô hình có thể nhận biết vật ở xa, ở gần camera và ảnh với nhiều resolution khác nhau, phù hợp cho Multi-scale training của YOLOv2
   
2. ResNet34: 
   Kiến trúc ResNet34 tận dụng Batch Normalization và Residual Connections giúp mô hình học cực kỳ hiệu quả những feature phức tạp, đồng thời giữ được long term dependencies. 
   Việc dùng Pre-trained weights từ ImageNet giúp phần bộ xương backbone chuyển giao fine-tune đặc trưng hình vát/góc/điểm cực kỳ rõ ràng và sắc nét. Việc sử dụng Batch Norm và Residual connection kết hợp mô hình pretrain giúp ích rất rõ mAP, khi tăng từ 0.0165 (ResNet50 - 40 epochs) lên tới 0.2016 (ResNet34 - 20 epochs)

3. Freeze / Unfreeze weights : 
   Mô hình có thêm hàm Freeze / Unfreeze để detection head (lớp fully connected cuối cùng) để mô hình có thể tập trung huấn luyện tập trung ở 10 epoch đầu tiên. Sau đó mới bắt đầu fine-tune backbone ResNet34 ở epoch 11 trở đi

4. Single stage detector :
   Mô hình chạy rất nhanh bởi tất cả từ trích xuất đặc trưng và phân loại đều được chạy từ một mạng CNN duy nhất và được tính cũng bằng một hàm loss duy nhất. Điều này tăng dáng kể khá năng Real Time detection của YOLOv1 khi mô hình đạt tới 45fps cao hơn so với video lúc bấy giờ (30fps)

---
## Nhược điểm kỹ thuật
1. Single class classification
   Việc mô hình YOLOv1 chỉ sử dụng cho bài toán single calss classification làm ảnh hưởng tới dự đoán các ảnh mà có nhiều hơn một object / label. Ngoài ra việc mô hình chỉ có thể predict được tối đa 98 bbx (2 bbx cho 1 cell) ảnh hưởng trực tiếp tới khả năng dự đoán multi-label objects. Chẳng hạn như một ảnh mà có nhiều người đứng thành một hàng thì mô hình sẽ vẽ một bounding box lớn bao hoàn toàn hàng mà không thể detet từng người một, thậm chí nếu mAP quá thấp (giống như repo này) thì mô hình còn không xuất ra bounding box. Một vấn để khác chính là việc YOLOv1 khó detect các object nhỏ (do thiếu bbx) và không có SPP (Spatial Pyramid Pooling) nhưng điều này đã được hỗ trơ một phần nhờ SE Block 

2. Hàm loss của YOLOv1
   Hàm loss cua3 YOLOv1 cực kì phức tạp và nhiệt điểm lớn nhất chính là việc sử dụng SME (sum squared error). Hơn thế, mô hình vẫn còn penalize các tọa độ tâm bbx lớn và bbx nhỏ gần như nhau cho dù tác giả đã sử dụng sqrt(w) và sqrt(h). Ngoài ra, với việc nhân hệ số lambda_coord = 5 sẽ penalize rất nặng các prediction sai, khiến hàm loss cực kì cao ở các epoch đầu tiên (5-10 epoch)
   
3. Kém linh hoạt với tỷ lệ khung hình mới (Aspect Ratios)
   YOLOv1 dự đoán trực tiếp tọa độ mà không sử dụng Anchor Boxes (như Faster R-CNN hay YOLOv2). Do đó, khi đối mặt với vật thể có tỷ lệ dài/rộng bất thường chưa có trong tập train, nó dễ phát hiện sai

4. Dữ liệu cho fine-tuning không quá nhiều
   Bài toán Object Detection không có quá nhiều label (trong Pascal VOC 2012 chỉ có 20 labels). Việc này dẫn đến các tham số của bộ lọc chưa được quá tối ưa và rất khó để co thể tự pretrain từ đầu. Ngoài ra, việc backbone ResNet34 được pretrain trên ảnh 224x224 và trực tiếp finet-tune 448x448 ảnh hưởng lớn tới các tham số của backbone do backbone chưa hề quen với các ảnh có resolution cao hơn

---
## Các cải tiến từ các bài báo sau này
Tuy mang tính đột phá về tốc độ, kiến trúc YOLOv1 vẫn tồn tại một số hạn chế nội tại:

1. Multi-label classification
   Mô hình YOLOv2 có nhiều bbx hơn (845) vì vậy có thể detect nhiều hơn các vật thể

2. Anchor box
   Việc sử dụng Anchor Box từ YOLOv2 trở đi giúp ích rất nhiều tới tài nguyên tính toán và mAP của mô hình khi thay vì phải tự predict toàn bộ bbx, mô hình chỉ cần dự đoán các offset từ các Anchor Box có sẵn. Sau này với YOLOv11, mô hình lại quay trở lại với predict hoàn toàn các bbx

3. Dữ liệu dồi dào hơn 
   Mô hình YOLOv2 sử dụng WordTree để có thể extend thêm từ bộ dữ liệu ImageNet để tăng số lượng label cho COCO

4. Data Augmentation
   Mô hình YOLOv1 trong repo không sử dụng tới Augmentation qua thư viện albumentations. Nếu bạn thích, bạn có thể thử nghiệm với ColorJitter và Horizontal/Vertical Flip để giúp mô hình fine-tune tốt hơn

5. Cơ chế attention
   SE Block trong repo là một cơ chế attention cực kì đơn giản và có thể extend hơn thành các cơ chế attention khác như của YOLOv5 kết hợp với Vision Transformer

5. Các cải tiến khác 
   Các cập nhật khác trong tương lai sẽ được tôi cập nhật khi tôi tìm hiểu tới các paper và tài liệu khác. Mong bạn sẽ đồng hành với tôi 

---
## Project Structure (Cấu trúc Thư mục)

```text
├── dataset.py            # Script xử lý PASCAL VOC Dataset và Dataloader
├── utils.py              # Xử lý NMS, mAP và Convert tọa độ Bounding Boxes
├── model.py              # YOLOv1 Architecture (ResNet50 + SE Block)
├── model_pretrained.py   # YOLOv1 Architecture (ResNet34 Pretrained + SE Block)
├── train.py              # Pipeline huấn luyện chính với AMP & Optimizer
├── predict.py            # Pipeline dự đoán ảnh thực tế trực tiếp
├── output_detection.jpg  # Ảnh kết quả minh họa (Prediction)
└── .gitignore            # Git ignore configs
```

---

## Quick Start

### 1. Cài đặt môi trường
Đảm bảo bạn đã cài đặt đủ các thư viện cần thiết trước khi bắt đầu:
```bash
pip install torch torchvision opencv-python tqdm albumentations
```

### 2. Huấn luyện mô hình (Training)
Để bắt đầu đào tạo mô hình từ đầu hoặc tiếp tục từ một Checkpoint có sẵn (`best_yolo_model.pth`):
```bash
python train.py
```

### 3. Suy luận (Predict / Inference)
Chạy mô hình trên một bức ảnh bất kỳ của bạn để xem khả năng nhận diện:
```bash
python predict.py --image your_image.jpg
```

<br>
<div align="center">
<i>Nếu các bạn thấy hữu ích, cho tôi xin một sao nhé. Cảm ơn các bạn đã ủng hộ! ⭐️</i>
</div>