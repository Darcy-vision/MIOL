import math
import numpy as np
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.hub import load_state_dict_from_url


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}


class ConvBlock(nn.Module):
    """
    Layer to perform a convolution followed by ELU
    """
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()

        self.conv = Conv3x3(in_channels, out_channels)
        self.nonlin = nn.ELU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.nonlin(out)
        return out


class Conv3x3(nn.Module):
    """
    Layer to pad and convolve input
    """
    def __init__(self, in_channels, out_channels, use_refl=True):
        super(Conv3x3, self).__init__()

        if use_refl:
            self.pad = nn.ReflectionPad2d(1)
        else:
            self.pad = nn.ZeroPad2d(1)
        self.conv = nn.Conv2d(int(in_channels), int(out_channels), 3)

    def forward(self, x):
        out = self.pad(x)
        out = self.conv(out)
        return out


def upsample(x):
    """
    Upsample input tensor by a factor of 2
    """
    return F.interpolate(x, scale_factor=2, mode="nearest")


def ConvLayer(x, params, name:str, downsample:bool):
    '''
    Params
    x:              输入特征图
    params:         网络权重
    name:           网络层名字(eg. 1, 2, 3, 4)
    downsample:     是否进行1x1卷积下采样
    '''
    # convblock1
    identity = x

    if downsample:
        out = F.conv2d(x, params[f'encoder.encoder.layer{name}.0.conv1.weight'], stride=2, padding=1)
    else:
        out = F.conv2d(x, params[f'encoder.encoder.layer{name}.0.conv1.weight'], padding=1)
    out = F.batch_norm(out, None, None, weight=params[f'encoder.encoder.layer{name}.0.bn1.weight'], 
                        bias=params[f'encoder.encoder.layer{name}.0.bn1.bias'], training=True)
    out = F.relu(out)

    out = F.conv2d(out, params[f'encoder.encoder.layer{name}.0.conv2.weight'], padding=1)
    out = F.batch_norm(out, None, None, weight=params[f'encoder.encoder.layer{name}.0.bn2.weight'], 
                        bias=params[f'encoder.encoder.layer{name}.0.bn2.bias'], training=True)

    if downsample:
        temp = F.conv2d(x, params[f'encoder.encoder.layer{name}.0.downsample.0.weight'], stride=2)
        identity = F.batch_norm(temp, None, None, weight=params[f'encoder.encoder.layer{name}.0.downsample.1.weight'], 
                                bias=params[f'encoder.encoder.layer{name}.0.downsample.1.bias'], training=True)

    out += identity
    x = F.relu(out)

    # conblock2
    identity = x

    out = F.conv2d(x, params[f'encoder.encoder.layer{name}.1.conv1.weight'], padding=1)
    out = F.batch_norm(out, None, None, weight=params[f'encoder.encoder.layer{name}.1.bn1.weight'], 
                        bias=params[f'encoder.encoder.layer{name}.1.bn1.bias'], training=True)
    out = F.relu(out)

    out = F.conv2d(out, params[f'encoder.encoder.layer{name}.1.conv2.weight'], padding=1)
    out = F.batch_norm(out, None, None, weight=params[f'encoder.encoder.layer{name}.1.bn2.weight'], 
                        bias=params[f'encoder.encoder.layer{name}.1.bn2.bias'], training=True)

    out += identity
    out = F.relu(out)

    return out


class ResNetImageInput(models.ResNet):
    """
    Constructs a resnet model with the input image concatenating random depth.
    Adapted from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    """

    def __init__(self, block, layers, num_classes=1000):
        super(ResNetImageInput, self).__init__(block, layers)
        self.inplanes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def ResnetImageModelInit(num_layers, pretrained=False):
    """
    Constructs a ResNet model.
    Args:
        num_layers (int): Number of resnet layers. Must be 18 or 50
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    assert num_layers in [18, 50], "Can only run with 18 or 50 layer resnet"
    blocks = {18: [2, 2, 2, 2], 50: [3, 4, 6, 3]}[num_layers]
    block_type = {18: models.resnet.BasicBlock,
                  50: models.resnet.Bottleneck}[num_layers]
    model = ResNetImageInput(block_type, blocks)

    if pretrained:
        loaded = load_state_dict_from_url(models.resnet.model_urls['resnet{}'.format(num_layers)])
        model.load_state_dict(loaded)
    return model


class PositionEncodingSine1DRelative(nn.Module):
    """
    relative sine encoding 1D, partially inspired by DETR (https://github.com/facebookresearch/detr)
    """

    def __init__(self, num_pos_feats=64, temperature=10000):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature

    @torch.no_grad()
    def forward(self, input):
        """
        :param inputs: NestedTensor
        :return: pos encoding [N,C,H,2W-1]
        """
        # update h and w if downsampling
        bs, _, h, w = input.size()

        # populate all possible relative distances
        x_embed = torch.linspace(0, w - 1, w, dtype=torch.float32, device=input.device)
            
        # follow the original Transformer pos_encoding
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=input.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, None] / dim_t  # 2W-1xC
        # interleave cos and sin instead of concatenate
        pos = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # 2W-1xC

        return pos


class STMDepthEncoder(nn.Module):
    """
    Pytorch module for a resnet encoder
    """
    def __init__(self, num_layers, pretrained):
        super(STMDepthEncoder, self).__init__()

        self.num_ch_enc = np.array([64, 64, 128, 256, 512])

        resnets = {18: models.resnet18,
                   34: models.resnet34,
                   50: models.resnet50,
                   101: models.resnet101,
                   152: models.resnet152}

        if num_layers not in resnets:
            raise ValueError(
                "{} is not a valid number of resnet layers".format(num_layers))

        self.encoder = ResnetImageModelInit(num_layers, pretrained)

        if num_layers > 34:
            self.num_ch_enc[1:] *= 4

    def forward(self, input_image):
        self.features = []
        x = self.encoder.conv1(input_image)
        x = self.encoder.bn1(x)
        self.features.append(self.encoder.relu(x))
        self.features.append(self.encoder.layer1(self.encoder.maxpool(self.features[-1])))
        self.features.append(self.encoder.layer2(self.features[-1]))
        self.features.append(self.encoder.layer3(self.features[-1]))
        self.features.append(self.encoder.layer4(self.features[-1]))

        return self.features

    def functional_forward(self, input_image, params):
        self.features = [] 
        x = F.conv2d(input_image, params['encoder.encoder.conv1.weight'], stride=2, padding=3)
        x = F.batch_norm(x, running_mean=None, running_var=None, weight=params['encoder.encoder.bn1.weight'], 
                            bias=params['encoder.encoder.bn1.bias'], training=True)
        self.features.append(F.relu(x))
        x = F.max_pool2d(self.features[-1], kernel_size=3, stride=2, padding=1)

        # layer1-4(2, 2down, 2down, 2down)
        self.features.append(ConvLayer(x, params, 1, False))
        self.features.append(ConvLayer(self.features[-1], params, 2, True))
        self.features.append(ConvLayer(self.features[-1], params, 3, True))
        self.features.append(ConvLayer(self.features[-1], params, 4, True))

        return self.features

# output 256*288
class STMDepthDecoder(nn.Module):
    """
    Pytorch module for a decoder
    """
    def __init__(self, num_ch_enc, max_disp, scales=range(5), num_output_channels=1, use_skips=True):
        super(STMDepthDecoder, self).__init__()

        self.alpha = max_disp
        

        self.num_output_channels = num_output_channels
        self.use_skips = use_skips
        self.upsample_mode = 'nearest'
        self.scales = scales

        self.num_ch_enc = num_ch_enc
        self.num_ch_dec = np.array([16, 32, 64, 128, 256])

        # decoder
        self.convs = OrderedDict()
        for i in range(4, -1, -1):
            # upconv_0
            num_ch_in = self.num_ch_enc[-1] if i == 4 else self.num_ch_dec[i + 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 0)] = ConvBlock(num_ch_in, num_ch_out)

            # upconv_1
            num_ch_in = self.num_ch_dec[i]
            if self.use_skips and i > 0:
                num_ch_in += self.num_ch_enc[i - 1]
            num_ch_out = self.num_ch_dec[i]
            self.convs[("upconv", i, 1)] = ConvBlock(num_ch_in, num_ch_out)

        for s in self.scales:
            self.convs[("dispconv", s)] = Conv3x3(self.num_ch_dec[s], self.num_output_channels)

        self.decoder = nn.ModuleList(list(self.convs.values()))
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_features):
        self.outputs = []

        # decoder
        x = input_features[-1]
        for i in range(4, 0, -1):
            x = self.convs[("upconv", i, 0)](x)
            x = [upsample(x)]
            if self.use_skips and i > 0:
                x += [input_features[i - 1]]
            x = torch.cat(x, 1)
            x = self.convs[("upconv", i, 1)](x)
            if i in self.scales:
                self.outputs.append(self.alpha * (self.sigmoid(self.convs[("dispconv", i)](x))))

        self.outputs = self.outputs[::-1]
        return self.outputs
    
    def functional_forward(self, input_features, params):
        self.outputs = []

        # decoder
        x = input_features[-1]
        for i in range(4):
            x = F.pad(x, pad=[1,1,1,1],mode='reflect')
            x = F.conv2d(x, params[f'decoder.decoder.{2*i}.conv.conv.weight'], params[f'decoder.decoder.{2*i}.conv.conv.bias'])
            x = F.elu(x)

            x = [upsample(x)]
            if self.use_skips and i < 4:
                x += [input_features[4 - 1 - i]]
            x = torch.cat(x, 1)

            x = F.pad(x, pad=[1,1,1,1],mode='reflect')
            x = F.conv2d(x, params[f'decoder.decoder.{2*i+1}.conv.conv.weight'], params[f'decoder.decoder.{2*i+1}.conv.conv.bias'])
            x = F.elu(x)

            if i in [1, 2, 3, 4]:
                temp = F.pad(x, pad=[1,1,1,1],mode='reflect')
                temp = F.conv2d(temp, params[f'decoder.decoder.{14-i}.conv.weight'], params[f'decoder.decoder.{14-i}.conv.bias'])
                self.outputs.append(self.alpha * (torch.sigmoid(temp)))

        self.outputs = self.outputs[::-1]
        return self.outputs
