import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from natten import NeighborhoodAttention1D
import torch.nn.functional as F
# from pthflops import count_ops
# from natten.flops import get_flops

"""
This code is CTAN.
This code is written by Jae-Man Shin.
The code reference is https://github.com/alexmehta/NAC-TCN-TCNs-with-Causal-NA

Reference:
Mehta, Alexander, and William Yang. "NAC-TCN: Temporal Convolutional Networks with Causal Dilated Neighborhood Attention for Emotion Understanding." Proceedings of the 2023 7th International Conference on Video and Image Processing. 2023.
"""

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock_Attention(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.0,local=14):
        super(TemporalBlock_Attention, self).__init__()
        self.k = kernel_size
        # print("local", local)
        self.dilation = dilation
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,stride=stride,padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout1d(dropout)
        # print(local + (dilation*(self.k-1)))
        self.conv2 = NeighborhoodAttention1D(dim = n_outputs, kernel_size = kernel_size, dilation = dilation, num_heads = 4)
        self.chomp2 = Chomp1d(padding)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout1d(dropout)
        self.net_1 = nn.Sequential(self.conv1,self.chomp1, self.elu1,self.dropout1)
        self.net_2 = nn.Sequential(self.conv2)
        self.net_3 = nn.Sequential(self.chomp2,self.elu2,self.dropout2)
        self.downsample = nn.Conv1d(
            n_inputs, n_outputs,1) if n_inputs != n_outputs else None
        self.elu = nn.ELU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)
    def forward(self, x):

        # print(x.shape)
        out = self.net_1(x)
        out = F.pad(out, (self.dilation*(self.k-1), 0))
        out = out.permute(0, 2, 1)
        out = self.net_2(out)
        out = out.permute(0, 2, 1)
        out = self.net_3(out)
        res = out + x
        return res, out

class CTAN(nn.Module):
    def __init__(self, input_ch = 1, num_inputs = 32, num_channels = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128], kernel_size=3, dropout=0.0, output_size = 1):
        super(CTAN, self).__init__()
        
        # First layer. This architecture is similar to ResNet first layer.
        self.convnet = nn.Conv1d(in_channels = 1, out_channels = 32, kernel_size=7, stride=2, bias=False)
        self.convnet2 = nn.Conv1d(in_channels = 32, out_channels = 128, kernel_size=1, stride=1, bias=False)
        
        # TCN network
        # layers = []
        # num_levels = len(num_channels)
        # for i in range(num_levels):
        #     dilation_size = 2 ** i
        #     in_channels = num_inputs if i == 0 else num_channels[i-1]
        #     out_channels = num_channels[i]
        #     layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
        #                              padding=(kernel_size-1) * dilation_size, dropout = dropout)]

        self.block1 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=1, padding=(kernel_size-1) * 1)
        self.block2 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=2, padding=(kernel_size-1) * 2)
        self.block3 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=4, padding=(kernel_size-1) * 4)
        self.block4 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=8, padding=(kernel_size-1) * 8)
        self.block5 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=16, padding=(kernel_size-1) * 16)
        
        self.block6 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=32, padding=(kernel_size-1) * 32)
        self.block7 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=64, padding=(kernel_size-1) * 64)
        self.block8 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=128, padding=(kernel_size-1) * 128)
        self.block9 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=256, padding=(kernel_size-1) * 256)
        self.block10 = TemporalBlock_Attention(128, 128, kernel_size=3, stride=1, dilation=512, padding=(kernel_size-1) * 512)

        # self.network = nn.Sequential(*layers)
        
        # Last layer for classification or regression.
        self.gmp = nn.AdaptiveMaxPool1d(1)
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.sig = nn.Sigmoid()
        self.elu0 = nn.ELU()
        self.clf = nn.Sequential(
                    nn.Linear(128, 128),
                    nn.ELU(),
                    nn.Linear(128, 128),
                    nn.ELU(),
                    nn.Linear(128, 128),
                    nn.ELU(),
                    nn.Linear(128, 1)
                    )

    def forward(self, x):
        x = self.convnet(x)
        x = self.convnet2(x)
        x, s1 = self.block1(x)
        x, s2 = self.block2(x)
        x, s3 = self.block3(x)
        x, s4 = self.block4(x)
        x, s5 = self.block5(x)
        
        x, s6 = self.block6(x)
        x, s7 = self.block7(x)
        x, s8 = self.block8(x)
        x, s9 = self.block9(x)
        x, s10 = self.block10(x)
        skip_connection = s1 + s2 + s3 + s4 + s5 + s6 + s7 + s8 + s9 + s10
        skip_connection = self.elu0(skip_connection)
        skip_connection = self.gap(skip_connection)
        skip_connection = skip_connection.view(skip_connection.size(0), -1)
        skip_connection = self.clf(skip_connection)
        outputs = self.sig(skip_connection)
        return outputs
