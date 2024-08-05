"""
Copy from https://github.com/chenyaofo/pytorch-cifar-models.
"""
import copy
import sys
import torch
import torch.nn as nn

from models.base_modular_model import BaseModularModel

try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from functools import partial
from typing import Union, List, Dict, Any, cast

cifar10_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg11_bn-eaeebf42.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg13_bn-c01e4a43.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg16_bn-6ee7ea24.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar10_vgg19_bn-57191229.pt',
}

cifar100_pretrained_weight_urls = {
    'vgg11_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg11_bn-57d0759e.pt',
    'vgg13_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg13_bn-5ebe5778.pt',
    'vgg16_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg16_bn-7d8c4031.pt',
    'vgg19_bn': 'https://github.com/chenyaofo/pytorch-cifar-models/releases/download/vgg/cifar100_vgg19_bn-b98f7bd7.pt',
}


class VGG(BaseModularModel):

    def __init__(
            self,
            features: nn.Module,
            classifier: nn.Module,
            init_weights: bool = True
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.classifier = classifier
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


conv_cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

classifier_cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [4096, 4096],  # as original VGG-16
    'B': [512, 512],
}


def make_conv_layers(cfg: List[Union[str, int]], batch_norm: bool = False) -> nn.Sequential:
    layers: List[nn.Module] = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_classifier_layers(cfg: List[Union[str, int]], input_dim, num_classes):
    layers: List[nn.Module] = []
    curr_input_dim = input_dim
    for v in cfg:
        v = cast(int, v)
        layers += [nn.Linear(curr_input_dim, v), nn.ReLU(inplace=True)]

        curr_input_dim = v

    layers += [nn.Linear(curr_input_dim, num_classes)]
    return nn.Sequential(*layers)


def _vgg(arch: str, cfg: str, batch_norm: bool,
         num_classes, model_urls: Dict[str, str],
         pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    if pretrained:
        kwargs['init_weights'] = False
    model = VGG(features=make_conv_layers(conv_cfgs[cfg], batch_norm=batch_norm),
                classifier=make_classifier_layers(classifier_cfgs["A"], conv_cfgs[cfg][-2], num_classes=num_classes),
                **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch],
                                              progress=progress)
        model.load_state_dict(state_dict)
    return model


def cifar10_vgg11_bn(*args, **kwargs) -> VGG: pass


def cifar10_vgg13_bn(*args, **kwargs) -> VGG: pass


def cifar10_vgg16_bn(*args, **kwargs) -> VGG: pass


def cifar10_vgg19_bn(*args, **kwargs) -> VGG: pass


def cifar100_vgg11_bn(*args, **kwargs) -> VGG: pass


def cifar100_vgg13_bn(*args, **kwargs) -> VGG: pass


def cifar100_vgg16_bn(*args, **kwargs) -> VGG: pass


def cifar100_vgg19_bn(*args, **kwargs) -> VGG: pass


thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for cfg, model_name in zip(["A", "B", "D", "E"], ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]):
        method_name = f"{dataset}_{model_name}"
        model_urls = cifar10_pretrained_weight_urls if dataset == "cifar10" else cifar100_pretrained_weight_urls
        num_classes = 10 if dataset == "cifar10" else 100
        setattr(
            thismodule,
            method_name,
            partial(_vgg,
                    arch=model_name,
                    cfg=cfg,
                    batch_norm=True,
                    model_urls=model_urls,
                    num_classes=num_classes)
        )
