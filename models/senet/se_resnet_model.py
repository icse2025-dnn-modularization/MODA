from torch import nn

from models.model_utils import get_model_leaf_layers
from models.resnet.resnet_model import BasicBlock, ResNet
from models.senet.se_layer import SELayer


class SEBasicBlock(BasicBlock):
    expansion = 1

    def __init__(self, *args, se_reduction=16, **kwargs):
        super().__init__(*args, **kwargs)
        self.se = SELayer(self.bn2.num_features, se_reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class SeResNet(ResNet):
    def get_list_of_layers_need_to_track_act(self):
        return [(name, l) for name, l in get_model_leaf_layers(self, return_with_layer_name=True) if
                isinstance(l, nn.ReLU) or isinstance(l, nn.Sigmoid)]


def _seresnet(block, layers, **kwargs):
    model = SeResNet(block, layers, **kwargs)
    return model


def SEResNet18(**kwargs):
    return _seresnet(SEBasicBlock, [2, 2, 2, 2], **kwargs)
