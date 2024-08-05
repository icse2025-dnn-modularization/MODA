from torch import nn

import __config_pythonpath
import argparse
import os
from collections import defaultdict

import numpy as np
import torch

from configs import ModelConfig
from dataset_loader import supported_repair_datasets, load_repair_dataset
from exp_repair.repair_models.combined_model import RepairedModel
from model_modularizer import compose_model_from_modular_masks, calculate_modular_layer_masks
from exp_repair.repair_models import create_weak_model, supported_weak_models, supported_strong_models
from model_trainer import train
from models import create_modular_model
from models.model_evaluation_utils import evaluate_model, evaluate_model_per_class, \
    collect_model_output_minmax_per_class, evaluate_model_in_confusion_matrix, plot_confusion_matrix
from models.model_utils import print_model_summary, get_runtime_device, redirect_stdout_to_file

DEVICE = get_runtime_device()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weak_model', type=str, choices=supported_weak_models, required=True)
    parser.add_argument('--strong_model', type=str, choices=supported_strong_models, required=True)
    parser.add_argument('--dataset', type=str, choices=supported_repair_datasets, required=True)
    parser.add_argument('--mixed_class', type=int, required=True)
    parser.add_argument('--repair_strategy', type=str, choices="moda, cnnsplitter", required=True)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataloader_num_workers', type=int, default=2)

    parser.add_argument('--target_epoch', type=int, required=True)

    args = parser.parse_args()

    return args


def load_a_module_from_strong_model(model_type, model_checkpoint_dir,
                                    train_loader, test_loader,
                                    orig_num_classes, module_class,
                                    activation_rate_threshold, add_norm_output_layer=False):
    assert isinstance(module_class, int)

    strong_model_checkpoint_path = os.path.join(model_checkpoint_dir, "strong_model.pt")
    strong_model_modular_mask_path = os.path.join(model_checkpoint_dir,
                                                  f"strong_model.pt.mod_mask.thres{activation_rate_threshold}.pt")
    strong_model = create_modular_model(model_type=model_type, num_classes=orig_num_classes,
                                        modular_training_mode=True)
    strong_model.load_pretrained_weights(strong_model_checkpoint_path)
    # print(evaluate_model_per_class(strong_model, test_loader, num_classes=orig_num_classes, device=DEVICE))

    if not os.path.exists(strong_model_modular_mask_path):
        calculate_modular_layer_masks(model=strong_model, data_loader=train_loader, num_classes=orig_num_classes,
                                      save_path=strong_model_modular_mask_path,
                                      activation_rate_threshold=activation_rate_threshold)

    module = compose_model_from_modular_masks(model_type, modular_model_params=strong_model.state_dict(),
                                              modular_masks_path=strong_model_modular_mask_path,
                                              orig_num_classes=orig_num_classes, target_classes=[module_class, ])

    if add_norm_output_layer:
        module = add_norm_output_layer_to_model(module, train_loader)
    return module


def add_norm_output_layer_to_model(model, data_loader):
    bound_values = collect_model_output_minmax_per_class(model=model, data_loader=data_loader,
                                                         num_classes=1, device=DEVICE)
    normed_model = lambda images: (model(images) - bound_values[:, 0]) / (
            bound_values[:, 1] - bound_values[:, 0])
    # to make model evaluable as "normal model"
    normed_model.eval = model.eval
    normed_model.to = model.to
    return normed_model


def repair_model_by_adding_calibration_layer(weak_model, strong_module, patch_output_index, calibration_data_loader,
                                             test_loader):
    repaired_model = RepairedModel(weak_model, strong_module, patch_output_index)
    repaired_model.to(DEVICE)

    _ = train(model=repaired_model,
              train_loader=calibration_data_loader, test_loader=test_loader,
              modular_params=defaultdict(float), learning_rate=0.05,
              num_epochs=20, checkpoint_every_n_epochs=-1,
              checkpoint_dir=None, tensorboard_writer=None)
    # torch.save(repaired_model.state_dict(), "./repair_weights/MC1.pt")

    return repaired_model


def repair_model_by_norm_output_ranges(weak_model, strong_module, patch_output_index):
    def aliased_repaired_model(images):
        weak_model_outputs = weak_model(images)
        strong_module_outputs = strong_module(images)

        final_outputs = weak_model_outputs
        final_outputs[:, patch_output_index] = strong_module_outputs.squeeze()
        return final_outputs

    def repaired_model_eval():
        weak_model.eval()
        strong_module.eval()

    aliased_repaired_model.eval = repaired_model_eval

    def repaired_model_to_device(device):
        weak_model.to(device)
        strong_module.to(device)

    aliased_repaired_model.to = repaired_model_to_device

    return aliased_repaired_model


def load_strong_module(args, add_norm_output_layer=False):
    # load strong model
    strong_model_type = args.strong_model
    # strong_model_type = "vgg16"  # args.model  # could be another
    # strong_model_type = "resnet18"  # args.model  # could be another
    # strong_model_type = args.model  # could be another
    num_classes, train_loader, test_loader = load_repair_dataset(for_model="strong",
                                                                 dataset_type=args.dataset,
                                                                 batch_size=args.batch_size,
                                                                 num_workers=args.dataloader_num_workers)
    strong_model_checkpoint_dir = os.path.join(ModelConfig.model_checkpoint_dir,
                                               f"{strong_model_type}_{args.dataset}"
                                               # f"/model__bs128__ep200__lr0.05/v_testRepair_strong")
                                               f"/model__bs128__ep200__lr0.05/v_testRepair_strong_mod")
    strong_module = load_a_module_from_strong_model(model_type=strong_model_type,
                                                    model_checkpoint_dir=strong_model_checkpoint_dir,
                                                    train_loader=train_loader,
                                                    test_loader=test_loader,
                                                    orig_num_classes=num_classes,
                                                    module_class=args.mixed_class,
                                                    activation_rate_threshold=0.9,
                                                    add_norm_output_layer=add_norm_output_layer)  # True for repair_model_by_norm_output_ranges
    return strong_module


def load_weak_model(args, add_norm_output_layer=False):
    # load weak model
    num_classes, weak_model_train_loader, _ = load_repair_dataset(for_model="weak",
                                                                  dataset_type=args.dataset,
                                                                  batch_size=args.batch_size,
                                                                  num_workers=args.dataloader_num_workers,
                                                                  mixed_class=args.mixed_class)
    weak_model_checkpoint_dir = os.path.join(ModelConfig.model_checkpoint_dir,
                                             f"{args.weak_model}_{args.dataset}"
                                             f"/model__bs128__ep200__lr0.05/v_testRepair_weak_mod")
    weak_model_checkpoint_path = os.path.join(weak_model_checkpoint_dir,
                                              f"weak_model.mc{args.mixed_class}.at_ep{args.target_epoch}.pt")
    # weak_model_checkpoint_path = os.path.join(weak_model_checkpoint_dir,
    #                                           f"weak_model.mc{args.mixed_class}.pt")
    weak_model = create_weak_model(model_type=args.weak_model, num_classes=num_classes)
    weak_model.load_pretrained_weights(weak_model_checkpoint_path)

    # weak_model = add_norm_output_layer_to_model(weak_model, weak_model_train_loader)
    if add_norm_output_layer:
        weak_model = nn.Sequential(
            weak_model,
            nn.Softmax(dim=1),
        )

    return weak_model


def main():
    args = get_args()
    # run_version = f"v{get_current_timestamp()}"
    # run_version = f"v_testRepair_std"
    # run_version = f"v_testRepair_mod"

    # args.mixed_class = 0
    # args.target_epoch = 190
    # args.target_epoch = 10
    num_classes, train_loader, test_loader = load_repair_dataset(for_model="weak",
                                                                 dataset_type=args.dataset,
                                                                 batch_size=args.batch_size,
                                                                 num_workers=args.dataloader_num_workers,
                                                                 mixed_class=args.mixed_class)
    print(f"\r[Dataset {args.dataset}]"
          f" Train Dim {train_loader.dataset.data.shape} "
          f"- Test Dim {test_loader.dataset.data.shape} "
          f"- Labels {test_loader.dataset.classes} "
          f"- Mixed_class: {args.mixed_class}")

    enable_cnnsplitter = args.repair_strategy == "cnnsplitter"
    weak_model = load_weak_model(args=args, add_norm_output_layer=enable_cnnsplitter)
    strong_module = load_strong_module(args=args, add_norm_output_layer=enable_cnnsplitter)

    # repair weak model by replacing with strong module
    if enable_cnnsplitter:
        repaired_model = repair_model_by_norm_output_ranges(weak_model=weak_model,
                                                            strong_module=strong_module,
                                                            patch_output_index=num_classes - 1)  # as we are fixing the last label
    else:
        repaired_model = repair_model_by_adding_calibration_layer(weak_model=weak_model,
                                                                  strong_module=strong_module,
                                                                  patch_output_index=num_classes - 1,
                                                                  calibration_data_loader=train_loader,
                                                                  test_loader=test_loader)
    for model_name, model in dict(weak_model=weak_model, repaired_model=repaired_model).items():
        test_acc = evaluate_model_per_class(model, test_loader, device=DEVICE,
                                            num_classes=num_classes, acc_in_percent=False,
                                            show_progress=False)
        print(f"\r[MC{args.mixed_class}-{model_name}]".ljust(25) +
              f"{round(test_acc[-1] * 100, 2)}",
              f"\t{round(np.mean(test_acc[:-1]) * 100, 2)}",
              f"\t{[round(c_acc * 100, 2) for c_acc in test_acc]}")


if __name__ == '__main__':
    main()
