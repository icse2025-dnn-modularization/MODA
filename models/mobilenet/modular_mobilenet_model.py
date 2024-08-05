import copy
from collections import defaultdict
from itertools import product

import torch
from torch import nn

from models.mobilenet.mobilenet_model import MobileNet, Block
from models.model_utils import get_model_leaf_layers
from models.modular_utils import clone_and_modify_layer_structure


class ModularBlock(Block):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def customize_block_layers_planes(self, modular_planes_cfg):
        self.conv1 = clone_and_modify_layer_structure(original_layer=self.conv1,
                                                      in_channels=modular_planes_cfg["conv1"]["out_dim"],
                                                      out_channels=modular_planes_cfg["conv1"]["out_dim"],
                                                      groups=modular_planes_cfg["conv1"]["out_dim"])
        self.bn1 = clone_and_modify_layer_structure(original_layer=self.bn1,
                                                    num_features=modular_planes_cfg["conv1"]["out_dim"])

        self.conv2 = clone_and_modify_layer_structure(original_layer=self.conv2,
                                                      in_channels=modular_planes_cfg["conv2"]["in_dim"],
                                                      out_channels=modular_planes_cfg["conv2"]["out_dim"])
        self.bn2 = clone_and_modify_layer_structure(original_layer=self.bn2,
                                                    num_features=modular_planes_cfg["conv2"]["out_dim"])

    def init_forward_pass_masks(self, modular_planes_cfg):
        # calculate mask to handle skip connection during forward pass
        conv1_in_mask = modular_planes_cfg["conv1"]["in_param_mask"]
        conv1_out_mask = modular_planes_cfg["conv1"]["out_param_mask"]

        # we can do this way because conv1_out_mask contains conv1_in_mask
        # check function: get_modular_model_cfg() - LINE: intersection_mask = l_in_param_mask & l_out_param_mask
        conv1_forward_mask = torch.zeros(conv1_out_mask.shape, dtype=torch.bool).cuda()
        conv1_forward_mask[conv1_in_mask] = True
        conv1_forward_mask = conv1_forward_mask[conv1_out_mask]
        self.conv1_forward_indices = conv1_forward_mask.nonzero().squeeze(1)

    def forward(self, x):
        transformed_x = torch.zeros(
            (x.shape[0], self.conv1.out_channels, x.shape[2], x.shape[3]),
            dtype=x.dtype, device=x.device
        )
        transformed_x[:, self.conv1_forward_indices] = x

        return super().forward(transformed_x)


class ModularMobileNet(MobileNet):
    def __init__(self, modular_cfg):
        block_cls = ModularBlock

        super().__init__(block_cls=block_cls)
        self._customize_modular_layers(modular_cfg=modular_cfg, block_cls=block_cls)

    def _customize_modular_layers(self, modular_cfg, block_cls):
        self.conv1 = clone_and_modify_layer_structure(original_layer=self.conv1,
                                                      in_channels=modular_cfg["conv1"]["_"]["in_dim"],
                                                      out_channels=modular_cfg["conv1"]["_"]["out_dim"])
        self.bn1 = clone_and_modify_layer_structure(original_layer=self.bn1,
                                                    num_features=modular_cfg["conv1"]["_"]["out_dim"])

        self._customize_modular_intermediate_blocks(modular_planes_cfg=modular_cfg["layers"],
                                                    block_cls=block_cls)

        self.linear = clone_and_modify_layer_structure(original_layer=self.linear,
                                                       in_features=modular_cfg["linear"]["_"]["in_dim"],
                                                       out_features=modular_cfg["linear"]["_"]["out_dim"])

    def _customize_modular_intermediate_blocks(self, modular_planes_cfg, block_cls):
        initial_mobilenet_blocks = [m for m in self.modules() if isinstance(m, block_cls)]

        for i, bl in enumerate(initial_mobilenet_blocks):
            block_i__modular_planes_cfg = {k.split(".", 1)[1]: v for k, v in modular_planes_cfg.items() if
                                           k.startswith(f"{i}.")}
            bl.customize_block_layers_planes(modular_planes_cfg=block_i__modular_planes_cfg)
            bl.init_forward_pass_masks(modular_planes_cfg=block_i__modular_planes_cfg)


def get_modular_model_cfg(modular_layer_mask_dict):
    modular_model_cfg = defaultdict(lambda: defaultdict(dict))
    l_in_param_mask = torch.ones(3, dtype=torch.bool)  # initial layer's input dim equals to image dim (RGB)
    last_layer = None
    for i, (l_name, l_mask) in enumerate(modular_layer_mask_dict.items()):
        if "." in l_name:
            parent_l_name, child_l_name = l_name.split(".", 1)
        else:
            parent_l_name, child_l_name = l_name, "_"

        l_out_param_mask = l_mask
        # handle group convolution layers
        if parent_l_name == "layers" and child_l_name.endswith("conv1"):
            intersection_mask = l_in_param_mask & l_out_param_mask
            if intersection_mask.sum() <= 0:
                # if there is no intersection, create a minimal (1-element) intersection mask
                intersection_mask = torch.zeros(l_out_param_mask.shape, dtype=l_out_param_mask.dtype,
                                                device=l_out_param_mask.device)
                first_true_index = l_out_param_mask.nonzero(as_tuple=True)[0][0]
                intersection_mask[first_true_index] = True

            l_in_param_mask = intersection_mask
            if last_layer:
                last_layer["out_param_mask"] = l_in_param_mask
                last_layer["out_dim"] = torch.sum(l_in_param_mask).item()

        modular_model_cfg[parent_l_name][child_l_name] = {"in_dim": torch.sum(l_in_param_mask).item(),
                                                          "out_dim": torch.sum(l_out_param_mask).item(),
                                                          "in_param_mask": l_in_param_mask,
                                                          "out_param_mask": l_out_param_mask, }
        last_layer = modular_model_cfg[parent_l_name][child_l_name]
        l_in_param_mask = l_out_param_mask

    return modular_model_cfg


def mapping_modular_activation_masks(modular_layer_masks, template_mobilenet_model):
    modular_supported_layer_names = [n for n, l in
                                     get_model_leaf_layers(template_mobilenet_model, return_with_layer_name=True) if
                                     isinstance(l, nn.Conv2d) or isinstance(l, nn.Linear)]
    assert len(modular_supported_layer_names) == len(modular_layer_masks)

    mapped_modular_layer_mask_dict = {k: v for k, v in zip(modular_supported_layer_names, modular_layer_masks)}

    return mapped_modular_layer_mask_dict


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
    curr_layer_name = "linear"
    in_param_mask = modular_model_cfg[curr_layer_name]["_"]["in_param_mask"]
    out_param_mask = modular_model_cfg[curr_layer_name]["_"]["out_param_mask"]
    for p_type in std_param_types:
        curr_param_name = f"{curr_layer_name}.{p_type}"
        modular_layer_params = modular_model_params[curr_param_name]
        if p_type == "weight":
            modular_layer_params = modular_layer_params[:, in_param_mask]
        modular_model_params[curr_param_name] = modular_layer_params[out_param_mask]

    # handle blocks
    for curr_layer_name, curr_layer_modular_cfg in modular_model_cfg.items():
        if not curr_layer_name == "layers":
            continue
        sub_layer_names = curr_layer_modular_cfg.keys()
        for sub_l_name in sub_layer_names:
            in_param_mask = modular_model_cfg[curr_layer_name][sub_l_name]["in_param_mask"]
            out_param_mask = modular_model_cfg[curr_layer_name][sub_l_name]["out_param_mask"]

            gen_layer_list = [sub_l_name, sub_l_name.replace("conv", "bn")]

            for prod_layer_name, prod_p_type in product(gen_layer_list, ext_param_types):
                prod_param_name = f"{curr_layer_name}.{prod_layer_name}.{prod_p_type}"
                if prod_param_name not in modular_model_params:
                    continue
                modular_layer_params = modular_model_params[prod_param_name]
                if prod_p_type == "weight" and len(modular_layer_params.shape) >= 2 \
                        and modular_layer_params.shape[1] == in_param_mask.shape[0]:
                    modular_layer_params = modular_layer_params[:, in_param_mask]
                modular_model_params[prod_param_name] = modular_layer_params[out_param_mask]

    return modular_model_params


def modular_mobilenet_generator(modular_layer_masks, model_params, modular_target_classes):
    modular_layer_mask_dict = mapping_modular_activation_masks(modular_layer_masks=modular_layer_masks,
                                                               template_mobilenet_model=MobileNet())
    modular_model_cfg = get_modular_model_cfg(modular_layer_mask_dict=modular_layer_mask_dict)
    modular_model = ModularMobileNet(modular_cfg=modular_model_cfg)
    # print_model_summary(modular_model)
    modular_model_params = get_modular_model_params(model_params=model_params,
                                                    modular_model_cfg=modular_model_cfg)
    modular_model.load_state_dict(modular_model_params)

    return modular_model
