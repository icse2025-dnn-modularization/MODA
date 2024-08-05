import copy
import itertools
import os
import random
from collections import defaultdict

import torch
from torch import nn

from configs import ModelConfig
from dataset_loader import load_dataset
from model_modularizer import calculate_modular_layer_masks, compose_model_from_modular_masks
from models.model_utils import print_model_summary, powerset, count_parameters, get_runtime_device, \
    get_model_leaf_layers
from models.modular_utils import get_activation_rate_during_inference
from models import create_modular_model, compose_model_from_module_masks
from models.model_evaluation_utils import evaluate_model

DEVICE = get_runtime_device()


def evaluate_modules(model_type, raw_model, modular_masks_path):
    softmax_layer = get_model_leaf_layers(raw_model, return_with_layer_name=False)[-1]
    assert isinstance(softmax_layer, nn.Linear)
    model_num_classes = softmax_layer.out_features
    model_param_count = count_parameters(raw_model)

    all_classes__modular_layer_masks = torch.load(modular_masks_path)
    # calculate_module_overlaps(raw_model, all_classes__modular_layer_masks)
    trackable_params = generate_trackable_params(raw_model.state_dict())
    modules = []
    module_total_sizes = []
    for curr_class in range(model_num_classes):
        curr_module = compose_model_from_modular_masks(model_type, trackable_params, modular_masks_path,
                                                       model_num_classes, [curr_class])
        modules.append(curr_module)
        curr_module_param_count = count_parameters(curr_module)
        module_total_sizes.append(curr_module_param_count / model_param_count)
        print(f"[Class {curr_class}] Module's param count: {curr_module_param_count:,} "
              f"({curr_module_param_count / model_param_count:.2f})")
    module_overlap_sizes = calculate_overlap_params(modules, model_param_count)
    # print(model_type)
    # print(raw_model)
    # print(modular_masks_path)
    # print(module_total_sizes)
    # print(module_overlap_sizes)
    print("module_total_size", sum(module_total_sizes) / len(module_total_sizes))
    print("module_overlap_sizes", sum(module_overlap_sizes) / len(module_overlap_sizes))


def generate_trackable_params(raw_model_params):
    # generate unique values to params to measure the overlap after those params are modularized to individual modules
    # this means the pretrained params will be replaced (so don't use it for evaluate accuracy of the model)

    trackable_model_params = {}
    unique_number = 0
    for param_name, params in raw_model_params.items():
        numel = params.numel()  # Total number of elements in the tensor
        new_tensor = torch.arange(unique_number, unique_number + numel, dtype=torch.float64)
        unique_number += numel
        trackable_model_params[param_name] = new_tensor.view(params.shape)

    return trackable_model_params


def calculate_overlap_params(modules, model_param_count):
    flatten_module_params = []
    for m in modules:
        flatten_param_set = set()
        for p in m.parameters():
            flatten_param_set.update(p.view(-1).tolist())
        flatten_module_params.append(flatten_param_set)

    # Calculate all combinations of modules
    indices_combinations = list(itertools.combinations(range(len(flatten_module_params)), 2))

    # 357 = sampling from population of [4950 indices_combinations] with confidence level of 95%, margin of error 5%
    # indices_combinations = random.sample(indices_combinations, k=357)

    print("indices_combinations", len(indices_combinations))
    overlap_sizes = []
    for module1_index, module2_index in indices_combinations:
        module1_params = flatten_module_params[module1_index]
        module2_params = flatten_module_params[module2_index]
        curr_intersection = len(module1_params & module2_params)
        # curr_union = len(module1_params | module2_params)
        overlap_sizes.append(curr_intersection / model_param_count)
    return overlap_sizes


def calculate_module_overlaps(raw_model, all_classes__modular_layer_masks):
    mask_split_lengths = [len(m) for m in all_classes__modular_layer_masks[0]]
    flatten_masks = torch.stack([torch.cat(class_mask)
                                 for class_mask in all_classes__modular_layer_masks.values()]).to(DEVICE)

    # Calculate all combinations of indices
    indices_combinations = list(itertools.combinations(range(len(flatten_masks)), 2))

    # Extract the masks for each combination
    mask1_indices, mask2_indices = zip(*indices_combinations)
    masks1 = torch.stack([flatten_masks[i] for i in mask1_indices])
    masks2 = torch.stack([flatten_masks[i] for i in mask2_indices])

    intersection = masks1 & masks2
    union = masks1 | masks2

    overlaps = []
    by_layer_intersection = []
    by_layer_union = []
    start_i = 0

    model_layer_dict = {l_name: l for l_name, l in get_model_leaf_layers(raw_model, return_with_layer_name=True)}
    for ms_len in mask_split_lengths:
        end_i = start_i + ms_len
        by_layer_jaccard_index = (intersection[:, start_i:end_i].sum(dim=1) /
                                  union[:, start_i:end_i].sum(dim=1)).mean().detach().item()
        overlaps.append(by_layer_jaccard_index)
        start_i = end_i

    return overlaps


def main():
    activation_rate_threshold = 0.1

    # dataset_type, num_classes = "svhn", 10
    dataset_type, num_classes = "cifar10", 10
    # dataset_type, num_classes = "cifar100", 100

    model_type = "vgg16"
    # model_type = "resnet18"
    # model_type = "mobilenet"

    checkpoint_path = f"./temp/{model_type}_{dataset_type}/model__bs128__ep200__lr0.05__aff1.0_dis1.0_comp0.3/model.pt"
    print(activation_rate_threshold, checkpoint_path)

    modular_masks_save_path = checkpoint_path + f".mod_mask.thres{activation_rate_threshold}.pt"

    raw_model = create_modular_model(model_type=model_type, num_classes=num_classes,
                                     modular_training_mode=True)

    evaluate_modules(model_type=model_type,
                     raw_model=raw_model,
                     modular_masks_path=modular_masks_save_path)


if __name__ == '__main__':
    main()
