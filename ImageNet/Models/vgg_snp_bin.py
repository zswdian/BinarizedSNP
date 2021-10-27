import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from BinarizedModules import BinSNPConv2d


__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
}


class VGG(nn.Module):

    def __init__(self, features, num_classes=1000):
        super(VGG, self).__init__()
        self.bn1 = nn.BatchNorm2d(3)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.features = features
        self.classifier = nn.Sequential(
            BinSNPConv2d(512 * 7 * 7, 4096, Linear=True, dropout=0.5),
            BinSNPConv2d(4096, 4096, Linear=True, dropout=0.5),
            nn.Linear(4096, num_classes),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv1(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg):
    layers = []
    in_channels = 64
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            binconv2d = BinSNPConv2d(in_channels, v, kernel_size=3, padding=1)
            layers += binconv2d
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': ['M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg11(pretrained=False, model_root=None, **kwargs):
    """VGG 11-layer model (configuration "A")"""
    model = VGG(make_layers(cfg['A']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg11'], model_root))
    return model


def vgg11_bn(**kwargs):
    """VGG 11-layer model (configuration "A") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['A'], batch_norm=True), **kwargs)


def vgg13(pretrained=False, model_root=None, **kwargs):
    """VGG 13-layer model (configuration "B")"""
    model = VGG(make_layers(cfg['B']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg13'], model_root))
    return model


def vgg13_bn(**kwargs):
    """VGG 13-layer model (configuration "B") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['B'], batch_norm=True), **kwargs)


def vgg16(pretrained=False, model_root=None, **kwargs):
    """VGG 16-layer model (configuration "D")"""
    model = VGG(make_layers(cfg['D']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg16'], model_root))
    return model


def vgg16_bn(**kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['D'], batch_norm=True), **kwargs)


def vgg19(pretrained=False, model_root=None, **kwargs):
    """VGG 19-layer model (configuration "E")"""
    model = VGG(make_layers(cfg['E']), **kwargs)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['vgg19'], model_root))
    return model


def vgg19_bn(**kwargs):
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    kwargs.pop('model_root', None)
    return VGG(make_layers(cfg['E'], batch_norm=True), **kwargs)