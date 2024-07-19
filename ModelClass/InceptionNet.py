import torch
import torch.nn as nn

"""
This code is InceptionNet code using 1-D convolution for biosignals.
This code is written by Jae-Man Shin.

Reference:
1. Szegedy, Christian, et al. "Going deeper with convolutions." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.
2. Park, Seong-A., et al. "Attention mechanisms for physiological signal deep learning: which attention should we take?." International Conference on Medical Image Computing and Computer-Assisted Intervention. Cham: Springer Nature Switzerland, 2022.
"""

class inception_block(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3, out_5x5_reduce, out_5x5, out_pool):
        super(inception_block, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_1x1, kernel_size = 1),
            nn.BatchNorm1d(out_1x1),
            nn.ReLU()
        )
        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_3x3_reduce, kernel_size = 1),
            nn.BatchNorm1d(out_3x3_reduce),
            nn.ReLU(),
            nn.Conv1d(out_3x3_reduce, out_3x3, kernel_size = 3, padding = 1),
            nn.BatchNorm1d(out_3x3),
            nn.ReLU()
        )
        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_5x5_reduce, kernel_size = 1),
            nn.BatchNorm1d(out_5x5_reduce),
            nn.ReLU(),
            nn.Conv1d(out_5x5_reduce, out_5x5, kernel_size = 5, padding = 2),
            nn.BatchNorm1d(out_5x5),
            nn.ReLU()
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm1d(out_pool),
            nn.ReLU()
        )
    def forward(self, x):
        out1x1 = self.branch1(x)
        out3x3 = self.branch2(x)
        out5x5 = self.branch3(x)
        out1x1pool = self.branch4(x)
        
        return torch.cat([out1x1, out3x3, out5x5, out1x1pool], 1)

class InceptionNet(nn.Module):
    def __init__(self, in_channels = 1):
        super(InceptionNet, self).__init__()        
        
        self.conv1 = nn.Conv1d(in_channels = in_channels, out_channels = 64, kernel_size=7, stride=2, padding=3)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv1d(in_channels = 64, out_channels = 192, kernel_size=3, stride=2, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.inception3a = inception_block(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = inception_block(256, 128, 128, 192, 32, 96, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.inception4a = inception_block(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = inception_block(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = inception_block(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = inception_block(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = inception_block(528, 256, 160, 320, 32, 128, 128)
        self.pool4 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.inception5a = inception_block(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = inception_block(832, 384, 192, 384, 48, 128, 128)
        
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.clf = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 512),
                    nn.ReLU(),
                    nn.Linear(512, 1),
                    nn.Sigmoid()
            )
    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.pool4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x
