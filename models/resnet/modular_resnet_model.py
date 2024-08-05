import copy
import json
import re
from collections import defaultdict
from itertools import product

import torch
from torch import nn

from models.model_utils import get_model_leaf_layers
from models.modular_utils import clone_and_modify_layer_structure
from models.resnet.resnet_model import ResNet, BasicBlock, conv1x1, conv3x3


class ModularBasicBlock(BasicBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def customize_block_layers_planes(self, modular_planes_cfg):
        # reinitialize only necessary layers after parent BasicBlock initialized above
        # this follows the current code of Resnet, and might not work for future version of Resnet model

        self.conv1 = clone_and_modify_layer_structure(original_layer=self.conv1,
                                                      in_channels=modular_planes_cfg["conv1"]["in_dim"],
                                                      out_channels=modular_planes_cfg["conv1"]["out_dim"])
        self.bn1 = clone_and_modify_layer_structure(original_layer=self.bn1,
                                                    num_features=modular_planes_cfg["conv1"]["out_dim"])

        if self.downsample is not None:
            assert "downsample.0" in modular_planes_cfg, "layer and masks mismatched"
            downsample_planes_cfg = modular_planes_cfg["downsample.0"]
            ds_conv2d_layer, ds_bn_layer = get_model_leaf_layers(self.downsample, return_with_layer_name=False)
            self.downsample = nn.Sequential(
                clone_and_modify_layer_structure(original_layer=ds_conv2d_layer,
                                                 in_channels=downsample_planes_cfg["in_dim"],
                                                 out_channels=downsample_planes_cfg["out_dim"]),
                clone_and_modify_layer_structure(original_layer=ds_bn_layer,
                                                 num_features=downsample_planes_cfg["out_dim"]),
            )

        self.conv2 = clone_and_modify_layer_structure(original_layer=self.conv2,
                                                      in_channels=modular_planes_cfg["conv2"]["in_dim"],
                                                      out_channels=modular_planes_cfg["conv2"]["out_dim"])
        self.bn2 = clone_and_modify_layer_structure(original_layer=self.bn2,
                                                    num_features=modular_planes_cfg["conv2"]["out_dim"])

    def init_forward_pass_masks(self, modular_planes_cfg):
        # calculate mask to handle skip connection during forward pass
        conv2_out_mask = modular_planes_cfg["conv2"]["out_param_mask"]
        if self.downsample is not None:
            # do nothing, currently "conv2" and "downsample" layers will have the same output dim
            # as they share the same output mask
            pass
        else:
            identity_out_mask = modular_planes_cfg["conv1"]["in_param_mask"]
            intersection_mask = conv2_out_mask & identity_out_mask

            self.conv2_forward_mask = intersection_mask[conv2_out_mask].nonzero().squeeze(1).cuda()
            self.identity_forward_mask = intersection_mask[identity_out_mask].nonzero().squeeze(1).cuda()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            out += identity
        else:
            out[:, self.conv2_forward_mask] += identity[:, self.identity_forward_mask]

        out = self.relu2(out)

        return out


class ModularResNet(ResNet):
    def __init__(self, modular_cfg, block, layers, **kwargs):
        super().__init__(block=block, layers=layers, **kwargs)
        self._customize_modular_layers(modular_cfg=modular_cfg, block_cls=block)

    def _customize_modular_layers(self, modular_cfg, block_cls):
        # reinitialize only necessary layers after parent BasicBlock initialized above
        self.conv1 = clone_and_modify_layer_structure(original_layer=self.conv1,
                                                      in_channels=modular_cfg["conv1"]["_"]["in_dim"],
                                                      out_channels=modular_cfg["conv1"]["_"]["out_dim"])
        self.bn1 = clone_and_modify_layer_structure(original_layer=self.bn1,
                                                    num_features=modular_cfg["conv1"]["_"]["out_dim"])

        self._customize_modular_intermediate_blocks(original_layer=self.layer1,
                                                    modular_planes_cfg=modular_cfg["layer1"],
                                                    block_cls=block_cls)
        self._customize_modular_intermediate_blocks(original_layer=self.layer2,
                                                    modular_planes_cfg=modular_cfg["layer2"],
                                                    block_cls=block_cls)
        self._customize_modular_intermediate_blocks(original_layer=self.layer3,
                                                    modular_planes_cfg=modular_cfg["layer3"],
                                                    block_cls=block_cls)
        self._customize_modular_intermediate_blocks(original_layer=self.layer4,
                                                    modular_planes_cfg=modular_cfg["layer4"],
                                                    block_cls=block_cls)

        self.fc = clone_and_modify_layer_structure(original_layer=self.fc,
                                                   in_features=modular_cfg["fc"]["_"]["in_dim"],
                                                   out_features=modular_cfg["fc"]["_"]["out_dim"])

    @staticmethod
    def _customize_modular_intermediate_blocks(original_layer, modular_planes_cfg, block_cls):
        initial_resnet_blocks = [m for m in original_layer.modules() if isinstance(m, block_cls)]

        for i, bl in enumerate(initial_resnet_blocks):
            block_i__modular_planes_cfg = {k.split(".", 1)[1]: v for k, v in modular_planes_cfg.items() if
                                           k.startswith(f"{i}.")}
            bl.customize_block_layers_planes(modular_planes_cfg=block_i__modular_planes_cfg)
            bl.init_forward_pass_masks(modular_planes_cfg=block_i__modular_planes_cfg)


def _modular_resnet(block, layers, modular_layer_masks, model_params, modular_target_classes, **kwargs):
    remapped_modular_layer_mask_dict \
        = mapping_modular_activation_masks(modular_layer_masks=modular_layer_masks,
                                           template_resnet_model=ResNet(block=block, layers=layers, **kwargs))
    modular_model_cfg = get_modular_model_cfg(modular_layer_mask_dict=remapped_modular_layer_mask_dict)
    modular_model = ModularResNet(modular_cfg=modular_model_cfg, block=block, layers=layers, **kwargs)
    # print_model_summary(modular_model)
    modular_model_params = get_modular_model_params(model_params=model_params,
                                                    modular_model_cfg=modular_model_cfg)
    modular_model.load_state_dict(modular_model_params)

    return modular_model


def mapping_modular_activation_masks(modular_layer_masks, template_resnet_model):
    """
         *this is special for Resnet model*
         since we have (unusual) RelU activation after the addition operation (conv2 + downsample) in Resnet Block
         and this means two layers share the same activation -> need to remap the masks to these layers
    """

    modular_supported_layer_names = [n for n, l in
                                     get_model_leaf_layers(template_resnet_model, return_with_layer_name=True) if
                                     isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)]

    mapped_modular_layer_mask_dict = {}
    act_layer_idx = 0
    for l_idx, l_name in enumerate(modular_supported_layer_names):
        if "0.downsample.0" in l_name:
            # downsample layer will share same activation mask with 0.conv2
            mapped_modular_layer_mask_dict[l_name] = modular_layer_masks[act_layer_idx - 1]
            continue
        mapped_modular_layer_mask_dict[l_name] = modular_layer_masks[act_layer_idx]
        act_layer_idx += 1

    return mapped_modular_layer_mask_dict


def get_modular_model_cfg(modular_layer_mask_dict):
    modular_model_cfg = defaultdict(lambda: defaultdict(dict))
    l_in_param_mask = torch.ones(3, dtype=torch.bool)  # initial layer's input dim equals to image dim (RGB)
    for i, (l_name, l_mask) in enumerate(modular_layer_mask_dict.items()):
        if "." in l_name:
            parent_l_name, child_l_name = l_name.split(".", 1)
        else:
            parent_l_name, child_l_name = l_name, "_"

        l_out_param_mask = l_mask

        if child_l_name == "0.downsample.0":
            l_in_param_mask = modular_model_cfg[parent_l_name]["0.conv1"]["in_param_mask"]

        modular_model_cfg[parent_l_name][child_l_name] = {"in_dim": torch.sum(l_in_param_mask).item(),
                                                          "out_dim": torch.sum(l_out_param_mask).item(),
                                                          "in_param_mask": l_in_param_mask,
                                                          "out_param_mask": l_out_param_mask, }
        l_in_param_mask = l_out_param_mask

    return modular_model_cfg


def get_modular_model_params(model_params, modular_model_cfg):
    modular_model_params = copy.deepcopy(model_params)
    std_param_types = {"weight", "bias"}
    ext_param_types = {*std_param_types, "running_mean", "running_var"}

    # handle (first) Conv layer
    curr_layer_name = "conv1"
    out_param_mask = modular_model_cfg[curr_layer_name]["_"]["out_param_mask"]
    for layer_name, p_type in product([curr_layer_name, curr_layer_name.replace("conv", "bn")], ext_param_types):
        curr_param_name = f"{layer_name}.{p_type}"
        if curr_param_name not in modular_model_params:
            continue
        modular_model_params[curr_param_name] = modular_model_params[curr_param_name][out_param_mask]

    # handle (last) FC layer
    curr_layer_name = "fc"
    in_param_mask = modular_model_cfg[curr_layer_name]["_"]["in_param_mask"]
    out_param_mask = modular_model_cfg[curr_layer_name]["_"]["out_param_mask"]
    for p_type in std_param_types:
        curr_param_name = f"{curr_layer_name}.{p_type}"
        modular_layer_params = modular_model_params[curr_param_name]
        if p_type == "weight":
            modular_layer_params = modular_layer_params[:, in_param_mask]
        modular_model_params[curr_param_name] = modular_layer_params[out_param_mask]

    # handle Resnet blocks
    for curr_layer_name, curr_layer_modular_cfg in modular_model_cfg.items():
        if not curr_layer_name.startswith("layer"):
            continue
        sub_layer_names = curr_layer_modular_cfg.keys()
        for sub_l_name in sub_layer_names:
            in_param_mask = modular_model_cfg[curr_layer_name][sub_l_name]["in_param_mask"]
            out_param_mask = modular_model_cfg[curr_layer_name][sub_l_name]["out_param_mask"]
            if "conv" in sub_l_name:
                gen_layer_list = [sub_l_name, sub_l_name.replace("conv", "bn")]
            else:
                # downsample layer
                gen_layer_list = [sub_l_name, sub_l_name[:-1] + "1"]
            for prod_layer_name, prod_p_type in product(gen_layer_list, ext_param_types):
                prod_param_name = f"{curr_layer_name}.{prod_layer_name}.{prod_p_type}"
                if prod_param_name not in modular_model_params:
                    continue
                modular_layer_params = modular_model_params[prod_param_name]
                if prod_p_type == "weight" and len(modular_layer_params.shape) >= 2:
                    modular_layer_params = modular_layer_params[:, in_param_mask]
                modular_model_params[prod_param_name] = modular_layer_params[out_param_mask]

    return modular_model_params


def modular_resnet18_generator(**kwargs):
    return _modular_resnet(block=ModularBasicBlock, layers=[2, 2, 2, 2], **kwargs)
