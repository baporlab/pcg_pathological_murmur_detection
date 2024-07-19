import torch
import torch.nn as nn

"""
This code is ResNet18 code using 1-D convolution for biosignals.
This code is written by Jae-Man Shin.

Reference:
1. He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
2. Park, Seong-A., et al. "Attention mechanisms for physiological signal deep learning: which attention should we take?." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2022.
"""


class residual_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(residual_block, self).__init__()
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = stride, padding = 1)
        self.conv2 = nn.Conv1d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1)
        self.batch1 = nn.BatchNorm1d(out_channels)
        self.batch2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()
        
        self.shortcut = nn.Sequential()
        if (stride != 1)|(in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size = 1, stride = stride),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x):
        res = self.shortcut(x)
        out = self.conv1(x)
        out = self.batch1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.batch2(out)
        out = out + res
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, in_channels = 1):
        super(ResNet, self).__init__()
        
        self.convnet = nn.Conv1d(in_channels = in_channels, out_channels = 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.batch = nn.BatchNorm1d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.res_block1 = residual_block(in_channels = 64, out_channels = 64, stride = 1)
        self.res_block2 = residual_block(in_channels = 64, out_channels = 64, stride = 1)
        
        self.res_block3 = residual_block(in_channels = 64, out_channels = 128, stride = 2)
        self.res_block4 = residual_block(in_channels = 128, out_channels = 128, stride = 1)
        
        self.res_block5 = residual_block(in_channels = 128, out_channels = 256, stride = 2)
        self.res_block6 = residual_block(in_channels = 256, out_channels = 256, stride = 1)
        self.res_block7 = residual_block(in_channels = 256, out_channels = 256, stride = 1)
        
        self.res_block8 = residual_block(in_channels = 256, out_channels = 512, stride = 2)
        self.res_block9 = residual_block(in_channels = 512, out_channels = 512, stride = 1)
        self.res_block10 = residual_block(in_channels = 512, out_channels = 512, stride = 1)
        
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.clf = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 1),
            nn.Sigmoid()
            )

    def forward(self, x):
        x = self.convnet(x)
        x = self.batch(x)
        x = self.relu(x)
        x = self.pool(x)

        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        x = self.res_block4(x)
        x = self.res_block5(x)
        x = self.res_block6(x)
        x = self.res_block7(x)
        x = self.res_block8(x)
        x = self.res_block9(x)
        x = self.res_block10(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x
