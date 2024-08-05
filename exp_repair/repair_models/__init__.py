from torch import nn

from models.lenet.lenet5_model import LeNet5
from models.vgg.vgg_model import cifar10_vgg16_bn as VGG16
from models.resnet.resnet_model import ResNet18

supported_weak_models = ["lenet5"]
supported_strong_models = ["vgg16", "resnet18"]


class PlaceholderBatchNorm2d(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x):
        return x


def create_weak_model(model_type, num_classes, modular_training_mode=False):
    if model_type == "vgg16":
        model = VGG16(num_classes=num_classes, batch_norm=False, pretrained=False)
    elif model_type == "resnet18":
        model = ResNet18(num_classes=num_classes, norm_layer=PlaceholderBatchNorm2d)
    elif model_type == "lenet5":
        model = LeNet5(num_classes=num_classes)
    else:
        raise Exception(f"model is not supported: {model_type}")

    if modular_training_mode:
        model.enable_hook_to_track_layer_activation_values()

    return model
