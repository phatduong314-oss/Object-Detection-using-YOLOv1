import os
import warnings

# Tắt cảnh báo update của Albumentations để không dính lỗi Time Out
os.environ["NO_ALBUMENTATIONS_UPDATE"] = "1"
warnings.filterwarnings("ignore", category=UserWarning, module="albumentations")

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchvision.transforms as transforms
import xml.etree.ElementTree as ET
import albumentations as A



S = 7
C = 20
IMAGE_SIZE = 448

VOC_classes = [
    'aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
    'bus', 'car', 'cat', 'chair', 'cow',
    'diningtable', 'dog', 'horse', 'motorbike', 'person',
    'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]

class VOC(Dataset):
    def __init__(self, root_dir, split='train', transform=None, image_size=IMAGE_SIZE, grid_size=S):
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.image_size = image_size
        self.grid_size = grid_size

        self.images_dir = os.path.join(root_dir, 'JPEGImages')
        self.annotations_dir = os.path.join(root_dir, 'Annotations')

        split_file = os.path.join(root_dir, 'ImageSets', 'Main', f'{split}.txt')
        with open(split_file, 'r') as f:
            self.image_ids = [line.strip() for line in f.readlines()]
        
        self.class_to_idx = {cls: idx for idx, cls in enumerate(VOC_classes)}

    def parse_annotation(self, annotation_path):
        tree = ET.parse(annotation_path)
        root = tree.getroot()
        size = root.find('size')
        image_width = float(size.find('width').text)
        image_height = float(size.find('height').text)
        
        annotations = []
        
        for obj in root.findall('object'):
            class_name = obj.find('name').text
            if class_name not in self.class_to_idx:
                continue
            
            # Get bounding box
            bbox = obj.find('bndbox')
            xmin = float(bbox.find('xmin').text)
            ymin = float(bbox.find('ymin').text)
            xmax = float(bbox.find('xmax').text)
            ymax = float(bbox.find('ymax').text)
            
            x = (xmin + xmax) / 2.0
            y = (ymin + ymax) / 2.0
            w = xmax - xmin
            h = ymax - ymin
            class_id = self.class_to_idx[class_name]
            
            annotations.append([x, y, w, h, class_id])
        
        return np.array(annotations, dtype=np.float32), image_width, image_height

    def encode_target(self, annotations, image_width, image_height):
        target = torch.zeros((self.grid_size, self.grid_size, C + 5), dtype=torch.float32)

        if annotations.size == 0:
            return target

        for x_center, y_center, width, height, class_id in annotations:
            x_center = float(x_center) / image_width
            y_center = float(y_center) / image_height
            width = float(width) / image_width
            height = float(height) / image_height

            grid_x = min(int(x_center * self.grid_size), self.grid_size - 1)
            grid_y = min(int(y_center * self.grid_size), self.grid_size - 1)

            if target[grid_y, grid_x, C] == 1:
                continue

            x_cell = x_center * self.grid_size - grid_x
            y_cell = y_center * self.grid_size - grid_y

            target[grid_y, grid_x, int(class_id)] = 1.0
            target[grid_y, grid_x, C] = 1.0
            target[grid_y, grid_x, C + 1:C + 5] = torch.tensor(
                [x_cell, y_cell, width, height],
                dtype=torch.float32,
            )

        return target

    def prepare_image(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))

        if self.transform:
            image = self.transform(image)
            if not isinstance(image, torch.Tensor):
                image = torch.as_tensor(image)
            return image.float()

        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        return image
    
    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        image_path = os.path.join(self.images_dir, f'{image_id}.jpg')
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        annotation_path = os.path.join(self.annotations_dir, f'{image_id}.xml')
        annotations, image_width, image_height = self.parse_annotation(annotation_path)
        target = self.encode_target(annotations, image_width, image_height)
        image = self.prepare_image(image)

        return image, target

    def __len__(self):
        return len(self.image_ids)
        
device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
train_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
    
train_data_dir = os.path.join('data', 'VOCdevkit', 'VOC2012')
train_dataset = VOC(train_data_dir, split='train', transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, num_workers=4, pin_memory=True)

test_dataset = VOC(train_data_dir, split='val', transform=test_transform)       
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)
    
