import torch
import torch.nn as nn

from ModelClass.VGG16 import *
from ModelClass.ResNet import *
from ModelClass.InceptionNet import *
from ModelClass.TCN import TCN
from ModelClass.EfficientNet import *
from ModelClass.CTAN import CTAN
from ModelClass.ConvNext import ConvNext

def load_model(model_name = 'vgg16', n_ch = 1):
    if model_name == 'vgg16':
        model = VGG16(in_channels = n_ch, out_channels = 64)
    elif model_name == 'resnet':
        model = ResNet(in_channels = n_ch)
    elif model_name == 'inception':
        model = InceptionNet(in_channels = n_ch)
    elif model_name == 'tcn':
        model = TCN(num_inputs = 32, num_channels = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128], kernel_size=3, dropout=0.0)
    elif model_name == 'ctan':
        model = CTAN(num_inputs = 32, num_channels = [128, 128, 128, 128, 128, 128, 128, 128, 128, 128], kernel_size=3, dropout=0.0)
    elif model_name == 'convnext':
        model = ConvNext(in_channels = n_ch)
        
    elif model_name == 'efficient':
        model = EfficientNet(num_leads = n_ch, num_classes = 1, model_num='efficientnet-b0')
            
    return model
