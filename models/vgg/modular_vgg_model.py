import copy
import sys
from functools import partial
from typing import Any

import torch

from models.vgg.vgg_model import VGG, make_conv_layers, make_classifier_layers, classifier_cfgs, conv_cfgs


def _modular_vgg(cfg: str, batch_norm: bool,
                 modular_layer_masks, model_params, modular_target_classes,
                 **kwargs: Any) -> VGG:
    kwargs['init_weights'] = False

    # **IMPORTANT**
    # make sure classes are sorted in incremental order (e.g., [0,1,2,3])
    # because module parameters are extracted based on this order
    # wrong order leads to wrong loaded params -> very low classification accuracy of extracted modules
    module_conv_cfg, module_classifier_cfg = get_modular_model_cfg(modular_layer_masks, conv_cfgs[cfg],
                                                                   classifier_cfgs["A"])
    modular_model = VGG(features=make_conv_layers(module_conv_cfg, batch_norm=batch_norm),
                        classifier=make_classifier_layers(module_classifier_cfg, module_conv_cfg[-2],
                                                          num_classes=len(modular_target_classes)),
                        **kwargs)
    modular_model_params = get_modular_model_params(model_params=model_params,
                                                    modular_layer_masks=modular_layer_masks)
    modular_model.load_state_dict(modular_model_params)

    return modular_model


def get_modular_model_cfg(modular_layer_masks, template_conv_cfg, classifier_cfg):
    new_conv_cfg = []
    new_classifier_cfg = []

    conv_index = 0
    modular_layer_masks = modular_layer_masks[:-1]  # not considering last (softmax) layer in the config
    conv_module_masks = modular_layer_masks[0:-len(classifier_cfg)]
    for item in template_conv_cfg:
        if not isinstance(item, int):
            new_conv_cfg.append(item)
        else:
            new_conv_cfg.append(torch.sum(conv_module_masks[conv_index]).item())
            conv_index += 1

    classifier_module_masks = modular_layer_masks[-len(classifier_cfg):]
    for classifier_index in range(len(classifier_cfg)):
        new_classifier_cfg.append(torch.sum(classifier_module_masks[classifier_index]).item())
    return new_conv_cfg, new_classifier_cfg


def get_modular_model_params(model_params, modular_layer_masks):
    modular_model_params = copy.deepcopy(model_params)

    visited_layer_set = set()
    for param_name, param_values in model_params.items():
        layer_name, param_type = param_name.rsplit(".", 1)  # e.g., features.35.running_var
        if param_type not in {"weight", "bias", "running_mean", "running_var"}:
            # print("Unsupported modular model with params:", param_name)
            continue
        if len(param_values.shape) >= 2:
            visited_layer_set.add(layer_name)

        curr_layer_index = len(visited_layer_set) - 1
        modular_layer_params = param_values

        # update num of kernels to fit input dim (start from 2nd layer and only apply to Conv or FC layers)
        if curr_layer_index > 0 and len(modular_layer_params.shape) >= 2:
            modular_layer_params = modular_layer_params[:, modular_layer_masks[curr_layer_index - 1]]

        # update num of kernels to fit output dim
        modular_layer_params = modular_layer_params[modular_layer_masks[curr_layer_index]]

        modular_model_params[param_name] = modular_layer_params

    return modular_model_params


def cifar10_modular_vgg11_bn(*args, **kwargs) -> VGG: pass


def cifar10_modular_vgg13_bn(*args, **kwargs) -> VGG: pass


def cifar10_modular_vgg16_bn(*args, **kwargs) -> VGG: pass


def cifar10_modular_vgg19_bn(*args, **kwargs) -> VGG: pass


def cifar100_modular_vgg11_bn(*args, **kwargs) -> VGG: pass


def cifar100_modular_vgg13_bn(*args, **kwargs) -> VGG: pass


def cifar100_modular_vgg16_bn(*args, **kwargs) -> VGG: pass


def cifar100_modular_vgg19_bn(*args, **kwargs) -> VGG: pass

thismodule = sys.modules[__name__]
for dataset in ["cifar10", "cifar100"]:
    for cfg, model_name in zip(["A", "B", "D", "E"], ["vgg11_bn", "vgg13_bn", "vgg16_bn", "vgg19_bn"]):
        method_name = f"{dataset}_{model_name}"
        num_classes = 10 if dataset == "cifar10" else 100

        method_name = f"{dataset}_modular_{model_name}"
        setattr(
            thismodule,
            method_name,
            partial(_modular_vgg,
                    cfg=cfg,
                    batch_norm=True)
        )
