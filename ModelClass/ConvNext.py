import torch
import torch.nn as nn
import torch.nn.functional as F

"""
ConvNext Block
This code is ConvNeXt model. We replaced residual block to ConvNeXt block.
The code reference is https://github.com/facebookresearch/ConvNeXt
This code is written by Jae-Man Shin.

Reference:
1. Liu, Zhuang, et al. "A convnet for the 2020s." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2022.

"""

class ConvNeXt_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3, groups = in_channels, stride = stride) # depthwise conv
        self.norm = nn.LayerNorm(out_channels, eps=1e-6)
        self.pwconv1 = nn.Linear(out_channels, 4 * out_channels) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * out_channels, out_channels)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((out_channels)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = nn.Identity()
        
        self.shortcut = nn.Sequential()
        if (stride != 1)|(in_channels != out_channels):
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size = 1, stride = stride),
                Permute(0, 2, 1),
                nn.LayerNorm(out_channels, eps=1e-6),
                Permute(0, 2, 1)
            )
        
    def forward(self, x):
        res = self.shortcut(x)
        out = self.dwconv(x)
        out = out.permute(0, 2, 1)
        out = self.norm(out)
        out = self.pwconv1(out)
        out = self.act(out)
        out = self.pwconv2(out)
        if self.gamma is not None:
            out = self.gamma * out
        out = out.permute(0, 2, 1)
        out = res + self.drop_path(out)
        return out
        
class ConvNext(nn.Module):
    def __init__(self, in_channels = 1):
        super(ConvNext, self).__init__()
        
        self.convnet = nn.Conv1d(in_channels = in_channels, out_channels = 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.layernorm = nn.LayerNorm(64)
        self.relu = nn.GELU()
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.res_block1 = ConvNeXt_block(in_channels = 64, out_channels = 64, stride = 1)
        self.res_block2 = ConvNeXt_block(in_channels = 64, out_channels = 64, stride = 1)
        
        self.res_block3 = ConvNeXt_block(in_channels = 64, out_channels = 128, stride = 2)
        self.res_block4 = ConvNeXt_block(in_channels = 128, out_channels = 128, stride = 1)
        
        self.res_block5 = ConvNeXt_block(in_channels = 128, out_channels = 256, stride = 2)
        self.res_block6 = ConvNeXt_block(in_channels = 256, out_channels = 256, stride = 1)
        self.res_block7 = ConvNeXt_block(in_channels = 256, out_channels = 256, stride = 1)
        
        self.res_block8 = ConvNeXt_block(in_channels = 256, out_channels = 512, stride = 2)
        self.res_block9 = ConvNeXt_block(in_channels = 512, out_channels = 512, stride = 1)
        self.res_block10 = ConvNeXt_block(in_channels = 512, out_channels = 512, stride = 1)
        
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
        x = x.permute(0, 2, 1)
        x = self.layernorm(x)
        x = x.permute(0, 2, 1)
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

class Permute(nn.Module):
    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(self.dims)
