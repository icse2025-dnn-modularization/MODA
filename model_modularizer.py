import argparse
import itertools
import os
import random
from collections import defaultdict

import torch

from configs import ModelConfig, BaseConfig
from dataset_loader import load_dataset, supported_std_datasets
from models.model_utils import print_model_summary, powerset, count_parameters, get_runtime_device
from models.modular_utils import get_activation_rate_during_inference
from models import create_modular_model, compose_model_from_module_masks, supported_models
from models.model_evaluation_utils import evaluate_model

DEVICE = get_runtime_device()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=supported_models, required=True)
    parser.add_argument('--dataset', type=str, choices=supported_std_datasets, required=True)

    parser.add_argument('--wf_affinity', type=float, default=1.0, help='weight factor for [modular] loss affinity')
    parser.add_argument('--wf_dispersion', type=float, default=1.0, help='weight factor for [modular] loss dispersion')
    parser.add_argument('--wf_compactness', type=float, default=0.3,
                        help='weight factor for [modular] loss compactness')

    parser.add_argument('--activation_rate_threshold', type=float, default=0.9, required=True)

    args = parser.parse_args()

    return args


@torch.no_grad()
def modularize_and_compose_model(model_type, modular_model, modular_masks_path,
                                 orig_num_classes, target_classes):
    target_classes = sorted(target_classes)
    composed_model = compose_model_from_modular_masks(model_type, modular_model.state_dict(), modular_masks_path,
                                                      orig_num_classes, target_classes)
    return composed_model


def evaluate_modularization_performance(std_model, modular_model, composed_model,
                                        test_loader, orig_num_classes, target_classes):
    eval_results = []
    model_list = [std_model, modular_model, composed_model]
    for i, curr_model in enumerate(model_list):
        curr_model.eval()
        model_acc = evaluate_model(curr_model, test_loader, device=DEVICE,
                                   num_classes=orig_num_classes, target_classes=target_classes,
                                   acc_in_percent=True, show_progress=False)
        model_param_count = count_parameters(curr_model)
        # print_model_summary(curr_model)
        eval_results.append({"acc": model_acc, "param_count": model_param_count})

    print(f"STD_MODEL_ACC: {eval_results[0]['acc']:.2f} ",
          f"- MOD_MODEL_ACC: {eval_results[1]['acc']:.2f} ",
          f"- COM_MODEL_ACC: {eval_results[2]['acc']:.2f} "
          f"(Params: {eval_results[2]['param_count']:,}/{eval_results[0]['param_count']:,}"
          f" ~ {eval_results[2]['param_count'] / eval_results[0]['param_count']:.2f}) "
          f"-------- {target_classes}")


def compose_model_from_modular_masks(model_type, modular_model_params, modular_masks_path, orig_num_classes,
                                     target_classes):
    all_classes__modular_layer_masks = torch.load(modular_masks_path)
    target_classes__modular_layer_masks = build_module_masks_for_target_classes(all_classes__modular_layer_masks,
                                                                                orig_num_classes, target_classes)
    composed_model = compose_model_from_module_masks(model_type, modular_model_params,
                                                     target_classes__modular_layer_masks,
                                                     target_classes)
    return composed_model


def build_module_masks_for_target_classes(all_classes__modular_layer_masks, orig_num_classes, target_classes):
    target_classes__modular_layer_masks = [all_classes__modular_layer_masks[tc] for tc in target_classes]
    merged_modular_layer_masks = [torch.any(torch.stack(l_mask), dim=0) for l_mask in
                                  zip(*target_classes__modular_layer_masks)]

    # handle last (softmax) FC layer:
    # instead of using activation values to decide keeping or removing a neuron (as other layers)
    # -> last layer's "neurons" will always be considered based on [target_classes]
    last_layer_mask = torch.zeros(orig_num_classes, dtype=torch.bool)
    last_layer_mask[target_classes] = True
    merged_modular_layer_masks.append(last_layer_mask)

    return merged_modular_layer_masks


def calculate_modular_layer_masks(model, data_loader, num_classes, save_path, activation_rate_threshold=0.9):
    all_classes__activation_rates, labels = get_activation_rate_during_inference(model,
                                                                                 data_loader=data_loader,
                                                                                 num_classes=num_classes,
                                                                                 device=DEVICE)
    all_classes__layer_masks = defaultdict(list)
    for each_class, all_layer_act_rates in all_classes__activation_rates.items():
        for layer_idx, layer_act_rate in all_layer_act_rates.items():
            valid_act = layer_act_rate >= activation_rate_threshold
            all_classes__layer_masks[each_class].append(valid_act.cpu())

    if not os.path.exists(save_path):
        torch.save(all_classes__layer_masks, save_path)
    else:
        print("File exists:", save_path)

    return all_classes__layer_masks


def generate_model_composition_tasks(num_classes):
    if num_classes == 10:
        all_classes = list(range(num_classes))
        for target_classes in list(powerset(all_classes)):
            if len(target_classes) <= 1:
                continue
            yield target_classes
    elif num_classes == 100:
        # if the file is not found, generate it by running the script exp_analysis/model_composition_sampler.py
        with open(os.path.join(BaseConfig.project_dir, "target_classes.num_classes_100.sample.list"), "r") as in_f:
            for line in in_f:
                yield [int(c) for c in line.strip().split()]
    else:
        raise NotImplementedError


def main():
    batch_size = 128
    args = get_args()
    print(args.__dict__)

    activation_rate_threshold = args.activation_rate_threshold
    dataset_type = args.dataset
    model_type = args.model
    model_checkpoint_dir = f"{ModelConfig.model_checkpoint_dir}/{model_type}_{dataset_type}/"

    mod_checkpoint_dir = os.path.join(model_checkpoint_dir,
                                      f"model__bs128__ep200__lr0.05__aff{args.wf_affinity}_dis{args.wf_dispersion}_comp{args.wf_compactness}")
    mod_checkpoint_path = [entry.path for entry in os.scandir(mod_checkpoint_dir)
                           if entry.is_dir() and entry.name.startswith("v")][-1] + "/model.pt"

    raw_checkpoint_dir = os.path.join(model_checkpoint_dir, f"model__bs128__ep200__lr0.05__aff0.0_dis0.0_comp0.0")
    raw_checkpoint_path = [entry.path for entry in os.scandir(raw_checkpoint_dir)
                           if entry.is_dir() and entry.name.startswith("v")][-1] + "/model.pt"

    print(f"\nModularization process started]\n"
          f"----\nStd_model_checkpoint_path: {raw_checkpoint_path}\n"
          f"----\nMod_model_checkpoint_path: {mod_checkpoint_path}\n"
          f"----\n")

    num_classes, train_loader, _ = load_dataset(dataset_type=dataset_type, batch_size=batch_size,
                                                num_workers=2)

    # load standard model
    std_model = create_modular_model(model_type=model_type, num_classes=num_classes,
                                     modular_training_mode=False)
    std_model.load_pretrained_weights(raw_checkpoint_path)
    print_model_summary(std_model)

    # load modular model
    mod_model = create_modular_model(model_type=model_type, num_classes=num_classes,
                                     modular_training_mode=True)
    mod_model.load_pretrained_weights(mod_checkpoint_path)

    # load modules
    modular_masks_save_path = mod_checkpoint_path + f".mod_mask.thres{activation_rate_threshold}.pt"
    if not os.path.exists(modular_masks_save_path):
        calculate_modular_layer_masks(model=mod_model, data_loader=train_loader, num_classes=num_classes,
                                      save_path=modular_masks_save_path,
                                      activation_rate_threshold=activation_rate_threshold)

    # if True:
    for target_classes in generate_model_composition_tasks(num_classes=num_classes):
        _, _, test_loader = load_dataset(dataset_type=dataset_type, target_classes=target_classes,
                                         batch_size=batch_size, num_workers=2)
        print(f"[Dataset {dataset_type}]- Test Dim {test_loader.dataset.data.shape}")
        composed_model = modularize_and_compose_model(model_type=model_type,
                                                      modular_model=mod_model,
                                                      modular_masks_path=modular_masks_save_path,
                                                      orig_num_classes=num_classes,
                                                      target_classes=target_classes)
        evaluate_modularization_performance(std_model=std_model,
                                            modular_model=mod_model,
                                            composed_model=composed_model,
                                            test_loader=test_loader,
                                            orig_num_classes=num_classes,
                                            target_classes=target_classes)


if __name__ == '__main__':
    main()
