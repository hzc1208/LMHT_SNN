import torch
import torch.nn as nn
from spikingjelly.clock_driven import layer


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('SpikingBasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in SpikingBasicBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = nn.ReLU(inplace=True)
        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out) + identity)

        return out


class MSBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(MSBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('MSBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in MSBlock")

        self.conv1 = layer.SeqToANNContainer(
            conv3x3(inplanes, planes, stride),
            norm_layer(planes)
        )
        self.sn1 = nn.ReLU(inplace=True)
        self.conv2 = layer.SeqToANNContainer(
            conv3x3(planes, planes),
            norm_layer(planes)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn2 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.conv1(self.sn1(x))
        out = self.conv2(self.sn2(out)) + identity

        return out
    

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        self.conv1 = layer.SeqToANNContainer(
            conv1x1(inplanes, width),
            norm_layer(width)
        )
        self.sn1 = nn.ReLU(inplace=True)

        self.conv2 = layer.SeqToANNContainer(
            conv3x3(width, width, stride, groups, dilation),
            norm_layer(width)
        )
        self.sn2 = nn.ReLU(inplace=True)

        self.conv3 = layer.SeqToANNContainer(
            conv1x1(width, planes * self.expansion),
            norm_layer(planes * self.expansion)
        )
        self.downsample = downsample
        self.stride = stride
        self.sn3 = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)
        out = self.sn1(self.conv1(x))
        out = self.sn2(self.conv2(out))
        out = self.sn3(self.conv3(out) + identity)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, use_MSBlock, num_classes=1000,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None, T=4, use_resnet19=False, use_dvs=False, use_imagenet=False):
        super(ResNet, self).__init__()
        self.T = T
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        self.use_imagenet = use_imagenet
        self.use_MSBlock = use_MSBlock
        self.use_dvs = use_dvs
        self.input_scale = 1.
        
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        if use_imagenet is True:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
            self.bn1 = norm_layer(self.inplanes)
        elif use_dvs is True:
            self.conv1 = layer.SeqToANNContainer(nn.Conv2d(2, self.inplanes, kernel_size=3, padding=1, bias=False))
            self.bn1 = layer.SeqToANNContainer(norm_layer(self.inplanes))
        else:
            self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=3, padding=1, bias=False)
            self.bn1 = norm_layer(self.inplanes)

        self.sn1 = nn.ReLU(inplace=True)
        self.maxpool = layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=3, stride=2, padding=1))
        self.avgpool = layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1)))
        
        if use_resnet19 is True:
            self.layer1 = nn.Sequential()
            self.layer2 = self._make_layer(block, 128, layers[0], stride=2 if use_dvs is True else 1,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.fc = nn.Sequential(
                layer.SeqToANNContainer(nn.Linear(512 * block.expansion, 256)),
                nn.ReLU(inplace=True),
                layer.SeqToANNContainer(nn.Linear(256, num_classes))
            )
        else:
            self.layer1 = self._make_layer(block, 64, layers[0])
            self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                           dilate=replace_stride_with_dilation[0])
            self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                           dilate=replace_stride_with_dilation[1])
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                           dilate=replace_stride_with_dilation[2])
            self.fc = layer.SeqToANNContainer(nn.Linear(512 * block.expansion, num_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                layer.SeqToANNContainer(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                ),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        if self.use_dvs is True:
            #x = x.transpose(0, 1)
            x = x.transpose(0, 1).mean(dim=0, keepdim=True)
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.sn1(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 2)
            return self.fc(x)            
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x.unsqueeze_(0)
            x = x * self.input_scale
            x = x.repeat(self.T, 1, 1, 1, 1)
            if self.use_MSBlock is False:
                x = self.sn1(x)
            if self.use_imagenet is True:
                x = self.maxpool(x)
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)

            x = self.avgpool(x)
            x = torch.flatten(x, 2)
            return self.fc(x)
        

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(block, layers, use_MSBlock, **kwargs):
    model = ResNet(block, layers, use_MSBlock, **kwargs)
    return model


def resnet18(**kwargs):
    return _resnet(BasicBlock, [2, 2, 2, 2], False, **kwargs)


def resnet19(**kwargs):
    return _resnet(BasicBlock, [3, 3, 2], False, **kwargs)


def resnet34(**kwargs):
    return _resnet(BasicBlock, [3, 4, 6, 3], False, **kwargs)


def ms_resnet18(**kwargs):
    return _resnet(MSBlock, [2, 2, 2, 2], True, **kwargs)
    
    
def ms_resnet34(**kwargs):
    return _resnet(MSBlock, [3, 4, 6, 3], True, **kwargs)


def resnet50(**kwargs):
    return _resnet(Bottleneck, [3, 4, 6, 3], False, **kwargs)


def resnet101(**kwargs):
    return _resnet(Bottleneck, [3, 4, 23, 3], False, **kwargs)


def resnet152(**kwargs):
    return _resnet(Bottleneck, [3, 8, 36, 3], False, **kwargs)
