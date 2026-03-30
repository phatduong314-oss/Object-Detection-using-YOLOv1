import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class BottleNeck(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*2, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channels*2)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels*2)  

    def forward(self, x):
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.se(x)  
        x = self.relu(x)

        return x
    
class YOLOV1(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
        resnet = models.resnet34(weights=weights, progress=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.transition = BottleNeck(in_channels=512, out_channels=512, stride=2)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(50176, 4096, bias=False),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 1470, bias=False),
            nn.Sigmoid() 
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.transition(x)
        x = self.head(x)
        x = x.reshape(-1, 7, 7, 30)

        return x
    
    def freeze_weights(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_weights(self):
        for param in self.backbone.parameters():
            param.requires_grad = True
    
if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOV1().to(device)
    model = model.to(memory_format=torch.channels_last)
    x = torch.randn((2, 3, 448, 448)).to(device) 
    out = model(x)
