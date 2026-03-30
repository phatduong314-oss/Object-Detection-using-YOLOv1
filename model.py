import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchinfo 

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
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.batchnorm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.batchnorm2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels*4, kernel_size=1, bias=False)
        self.batchnorm3 = nn.BatchNorm2d(out_channels*4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels * 4)  
        self.downsample = downsample

    def forward(self, x):
        identity = x
        x = self.conv1(x)
        x = self.batchnorm1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = self.se(x)  
        
        if self.downsample is not None : 
            identity = self.downsample(identity)

        x += identity
        x = self.relu(x)

        return x
    
class YOLOV1(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        self.block1 = self._block(BottleNeck, 64, 64, 3, 1)
        self.block2 = self._block(BottleNeck, 256, 128, 4, 2)
        self.block3 = self._block(BottleNeck, 512, 256, 6, 2)
        self.block4 = self._block(BottleNeck, 1024, 512, 3, 2) #2048x14x14
        self.conv = nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1, bias=False) #1024x7x7
        self.final = nn.Sequential(
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1024),
            nn.SiLU(inplace=True),            
            nn.Dropout(0.5),            
            nn.Conv2d(1024, 30, kernel_size=1),
            nn.Sigmoid()
        )

    def _block(self, block, in_channels, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels * 4 :
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels*4, kernel_size=1, stride = stride, bias=False),
                nn.BatchNorm2d(out_channels*4)
            )

        blocks = []
        blocks.append(block(in_channels, out_channels, stride, downsample))
        in_channels = out_channels * 4

        for _ in range(1, num_blocks):
            blocks.append(block(in_channels, out_channels, stride=1))

        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.conv(x)
        x = self.final(x)
        x = x.permute(0, 2, 3, 1)

        return x

if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = YOLOV1().to(device)
    model = model.to(memory_format=torch.channels_last)
    x = torch.randn((2, 3, 448, 448)).to(device) 
    out = model(x)
    print(out)