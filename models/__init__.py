from models.lenet.lenet5_model import LeNet5

from models.mobilenet.mobilenet_model import MobileNet
from models.mobilenet.modular_mobilenet_model import modular_mobilenet_generator as ModularMobileNet

from models.vgg.vgg_model import cifar10_vgg16_bn as VGG16
from models.vgg.modular_vgg_model import cifar10_modular_vgg16_bn as ModularVGG16

from models.resnet.resnet_model import ResNet18
from models.resnet.modular_resnet_model import modular_resnet18_generator as ModularResNet18

from models.senet.se_resnet_model import SEResNet18

supported_models = ["vgg16", "resnet18", "mobilenet"]


def create_modular_model(model_type, num_classes, modular_training_mode=False):
    if model_type == "vgg16":
        model = VGG16(num_classes=num_classes, pretrained=False)
    elif model_type == "resnet18":
        model = ResNet18(num_classes=num_classes)
    elif model_type == "mobilenet":
        model = MobileNet(num_classes=num_classes)
    elif model_type == "senet":
        model = SEResNet18(num_classes=num_classes)
    elif model_type == "lenet5":
        model = LeNet5(num_classes=num_classes)
    else:
        raise Exception(f"model is not supported: {model_type}")

    if modular_training_mode:
        model.enable_hook_to_track_layer_activation_values()

    return model


def compose_model_from_module_masks(model_type, raw_model_params, modular_layer_masks, target_classes):
    if model_type == "vgg16":
        modular_model = ModularVGG16(modular_layer_masks=modular_layer_masks,
                                     model_params=raw_model_params, modular_target_classes=target_classes)
    elif model_type == "resnet18":
        modular_model = ModularResNet18(modular_layer_masks=modular_layer_masks,
                                        model_params=raw_model_params, modular_target_classes=target_classes)
    elif model_type == "mobilenet":
        modular_model = ModularMobileNet(modular_layer_masks=modular_layer_masks,
                                         model_params=raw_model_params, modular_target_classes=target_classes)
    else:
        raise Exception(f"model is not supported: {model_type}")

    return modular_model
