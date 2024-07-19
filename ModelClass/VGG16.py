import torch
import torch.nn as nn

"""
This code is VGG16 code using 1-D convolution for biosignals.
This code is written by Jae-Man Shin.
This model does not use "flatten layer" after 5 convolution blocks.

Reference:
1. Simonyan, Karen, and Andrew Zisserman. "Very deep convolutional networks for large-scale image recognition." arXiv preprint arXiv:1409.1556 (2014).
"""

class VGG16(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 64):
        super(VGG16, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv1d(in_channels = in_channels, out_channels = out_channels, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            nn.Conv1d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(),
            )
        self.block2 = nn.Sequential(
            nn.Conv1d(in_channels = out_channels, out_channels = out_channels*2, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channels*2),
            nn.ReLU(),
            nn.Conv1d(in_channels = out_channels*2, out_channels = out_channels*2, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(out_channels*2),
            nn.ReLU(),
            )
        self.block3 = nn.Sequential(
            nn.Conv1d(in_channels = out_channels*2, out_channels = out_channels*4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channels*4),
            nn.ReLU(),
            nn.Conv1d(in_channels = out_channels*4, out_channels = out_channels*4, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channels*4),
            nn.ReLU(),
            nn.Conv1d(in_channels = out_channels*4, out_channels = out_channels*4, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(out_channels*4),
            nn.ReLU(),
            )
        self.block4 = nn.Sequential(
            nn.Conv1d(in_channels = out_channels*4, out_channels = out_channels*8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channels*8),
            nn.ReLU(),
            nn.Conv1d(in_channels = out_channels*8, out_channels = out_channels*8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channels*8),
            nn.ReLU(),
            nn.Conv1d(in_channels = out_channels*8, out_channels = out_channels*8, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(out_channels*8),
            nn.ReLU(),
            )
        self.block5 = nn.Sequential(
            nn.Conv1d(in_channels = out_channels*8, out_channels = out_channels*8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channels*8),
            nn.ReLU(),
            nn.Conv1d(in_channels = out_channels*8, out_channels = out_channels*8, kernel_size = 3, stride = 1, padding = 1),
            nn.BatchNorm1d(out_channels*8),
            nn.ReLU(),
            nn.Conv1d(in_channels = out_channels*8, out_channels = out_channels*8, kernel_size = 3, stride = 2, padding = 1),
            nn.BatchNorm1d(out_channels*8),
            nn.ReLU(),
            )
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.clf = nn.Sequential(
            nn.Linear(out_channels*8, out_channels*8),
            nn.ReLU(),
            nn.Linear(out_channels*8, out_channels*8),
            nn.ReLU(),
            nn.Linear(out_channels*8, out_channels*8),
            nn.ReLU(),
            nn.Linear(out_channels*8, 1),
            nn.Sigmoid()
            )
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.clf(x)
        return x
