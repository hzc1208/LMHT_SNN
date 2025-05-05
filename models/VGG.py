from unicodedata import numeric
import torch.nn as nn
import torch
from spikingjelly.clock_driven import layer


cfg = {
    'VGG11': [
        [64, 'M'],
        [128, 'M'],
        [256, 256, 'M'],
        [512, 512, 'M'],
        [512, 512, 'M']
    ],
    'VGG13': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 'MM']
    ],
    'VGG16': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 'M'],
        [512, 512, 512, 'M'],
        [512, 512, 512, 'M']
    ],
    'VGG19': [
        [64, 64, 'M'],
        [128, 128, 'M'],
        [256, 256, 256, 256, 'M'],
        [512, 512, 512, 512, 'M'],
        [512, 512, 512, 512, 'M']
    ],
    'CIFARNet': [
        [128],
        [256, 'M'],
        [512, 'M'],
        [1024],
        [512]
    ]
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, dropout, T, init_c=3):
        super(VGG, self).__init__()
        self.init_channels = init_c
        self.T = T
        self.vgg_name = vgg_name
        self.layer1 = self._make_layers(cfg[vgg_name][0], dropout)
        self.layer2 = self._make_layers(cfg[vgg_name][1], dropout)
        self.layer3 = self._make_layers(cfg[vgg_name][2], dropout)
        self.layer4 = self._make_layers(cfg[vgg_name][3], dropout)
        self.layer5 = self._make_layers(cfg[vgg_name][4], dropout)
        
        if vgg_name == 'CIFARNet':
            self.classifier = nn.Sequential(
                layer.SeqToANNContainer(nn.Linear(512*8*8, 1024)),
                nn.ReLU(inplace=True),
                layer.SeqToANNContainer(nn.Dropout(dropout)),
                layer.SeqToANNContainer(nn.Linear(1024, 512)),
                nn.ReLU(inplace=True),
                layer.SeqToANNContainer(nn.Dropout(dropout)),
                layer.SeqToANNContainer(nn.Linear(512, num_classes))
            )          
        elif num_classes == 1000:
            self.classifier = nn.Sequential(
                layer.SeqToANNContainer(nn.Linear(512*7*7, 4096)),
                nn.ReLU(inplace=True),
                layer.SeqToANNContainer(nn.Dropout(dropout)),
                layer.SeqToANNContainer(nn.Linear(4096, 4096)),
                nn.ReLU(inplace=True),
                layer.SeqToANNContainer(nn.Dropout(dropout)),
                layer.SeqToANNContainer(nn.Linear(4096, num_classes))
            )         
        elif num_classes == 200:
            self.classifier = layer.SeqToANNContainer(nn.Linear(512, num_classes))  
        else:
            self.classifier = nn.Sequential(
                layer.SeqToANNContainer(nn.Linear(512, 4096)),
                nn.ReLU(inplace=True),
                layer.SeqToANNContainer(nn.Dropout(dropout)),
                layer.SeqToANNContainer(nn.Linear(4096, 4096)),
                nn.ReLU(inplace=True),
                layer.SeqToANNContainer(nn.Dropout(dropout)),
                layer.SeqToANNContainer(nn.Linear(4096, num_classes))
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.zeros_(m.bias)

    def _make_layers(self, cfg, dropout):
        layers = []
        for x in cfg:
            if x == 'M':
                #layers.append(layer.SeqToANNContainer(nn.MaxPool2d(kernel_size=2, stride=2)))
                layers.append(layer.SeqToANNContainer(nn.AvgPool2d(kernel_size=2, stride=2)))
            elif x == 'MM':
                layers.append(layer.SeqToANNContainer(nn.AdaptiveAvgPool2d((1, 1))))
            else:
                layers.append(layer.SeqToANNContainer(
                    nn.Conv2d(self.init_channels, x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(x)
                ))
                layers.append(nn.ReLU(inplace=True))
                layers.append(layer.SeqToANNContainer(nn.Dropout(dropout)))
                self.init_channels = x
        return nn.Sequential(*layers)

    def forward(self, x):

        x = x.unsqueeze_(0).repeat(self.T, 1, 1, 1, 1)
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = torch.flatten(out, 2)
        out = self.classifier(out)
        return out


def vgg11(num_classes, time_step, dropout=0):
    return VGG('VGG11', num_classes, dropout, time_step)


def vgg13(num_classes, time_step, dropout=0):
    return VGG('VGG13', num_classes, dropout, time_step)


def vgg16(num_classes, time_step, dropout=0):
    return VGG('VGG16', num_classes, dropout, time_step)


def vgg19(num_classes, time_step, dropout=0):
    return VGG('VGG19', num_classes, dropout, time_step)
    

def cifarnet(num_classes, time_step, dropout=0):
    return VGG('CIFARNet', num_classes, dropout, time_step)
