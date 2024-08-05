import argparse
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from configs import ModelConfig
from dataset_loader import load_dataset, supported_std_datasets
from models import create_modular_model, supported_models
from models.model_evaluation_utils import evaluate_model
from models.model_utils import print_model_summary, get_runtime_device, redirect_stdout_to_file, get_current_human_time, \
    get_current_timestamp
from models.modular_utils import calculate_module_losses, mean_list

DEVICE = get_runtime_device()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=supported_models, required=True)
    parser.add_argument('--dataset', type=str, choices=supported_std_datasets, required=True)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataloader_num_workers', type=int, default=2)

    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=-1)

    parser.add_argument('--wf_affinity', type=float, default=0.0, help='weight factor for [modular] loss affinity')
    parser.add_argument('--wf_dispersion', type=float, default=0.0, help='weight factor for [modular] loss dispersion')
    parser.add_argument('--wf_compactness', type=float, default=0.0,
                        help='weight factor for [modular] loss compactness')

    args = parser.parse_args()

    return args


def train(model, train_loader, test_loader,
          modular_params, learning_rate, num_epochs, checkpoint_every_n_epochs,
          checkpoint_dir, checkpoint_model_name="model", tensorboard_writer=None):
    print(f"\r[{get_current_human_time()}][START TRAINING] Model checkpoint dir: {checkpoint_dir}")

    model.to(DEVICE)
    optimizer = torch.optim.SGD(
        params=model.parameters(), lr=learning_rate, momentum=0.9, nesterov=True
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    for curr_epoch in range(1, num_epochs + 1):
        train_acc, loss_value_dict = train_one_epoch(model, train_loader,
                                                     modular_params=modular_params,
                                                     optimizer=optimizer,
                                                     tqdm_desc=f"Epoch {curr_epoch:02}/{num_epochs}")

        test_acc = evaluate_model(model, test_loader, device=DEVICE, show_progress=False) \
            if test_loader else float('nan')

        scheduler.step()

        if checkpoint_dir is not None:
            if curr_epoch == num_epochs:
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, f"{checkpoint_model_name}.pt"))
            elif checkpoint_every_n_epochs > 0 and curr_epoch % checkpoint_every_n_epochs == 0:
                torch.save(model.state_dict(),
                           os.path.join(checkpoint_dir, f"{checkpoint_model_name}.at_ep{curr_epoch}.pt"))

        loss_log_str = "[" + " | ".join([f"{k}: {v:.2f}" for k, v in loss_value_dict.items()]) + "]"
        print(f"\r[{get_current_human_time()}] train_acc: {train_acc:.2f} - test_acc: {test_acc:.2f} - {loss_log_str}")

        if tensorboard_writer:
            tensorboard_writer.add_scalar(f'Train/Accuracy', train_acc, curr_epoch)
            tensorboard_writer.add_scalar(f'Test/Accuracy', test_acc, curr_epoch)
            for l_name, l_value in loss_value_dict.items():
                tensorboard_writer.add_scalar(f"Train/{l_name}", l_value, curr_epoch)

    return model


def train_one_epoch(model, train_loader, modular_params, optimizer, tqdm_desc=""):
    model.train()

    wf_loss_affinity = modular_params["wf_affinity"]
    wf_loss_dispersion = modular_params["wf_dispersion"]
    wf_loss_compactness = modular_params["wf_compactness"]
    modular_model_training_mode = any([wf_loss_affinity,
                                       wf_loss_dispersion,
                                       wf_loss_compactness])

    pbar = tqdm(train_loader, desc=tqdm_desc)
    correct, total = 0, 0
    all_loss_overall, all_loss_ce, all_loss_affinity, \
        all_loss_dispersion, all_loss_compactness = [], [], [], [], []

    for i, (images, labels) in enumerate(pbar):
        images, labels = images.to(DEVICE), labels.to(DEVICE)
        outputs = model(images)

        curr_loss_ce = F.cross_entropy(outputs, labels)
        if modular_model_training_mode:
            modular_activations = model.get_modular_activations_of_current_batch()
            curr_loss_affinity, curr_loss_dispersion, \
                curr_loss_compactness = calculate_module_losses(modular_activations, labels)
            curr_loss = curr_loss_ce + \
                        (wf_loss_affinity * curr_loss_affinity +
                         wf_loss_dispersion * curr_loss_dispersion +
                         wf_loss_compactness * curr_loss_compactness)
        else:
            curr_loss = curr_loss_ce

        optimizer.zero_grad()
        curr_loss.backward()
        optimizer.step()

        total += labels.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum()

        all_loss_overall.append(curr_loss.detach())
        all_loss_ce.append(curr_loss_ce.detach())

        if modular_model_training_mode:
            all_loss_affinity.append(curr_loss_affinity.detach())
            all_loss_dispersion.append(curr_loss_dispersion.detach())
            all_loss_compactness.append(curr_loss_compactness.detach())

    train_acc = (correct / total).item()

    loss_value_dict = {"loss_overall": mean_list(all_loss_overall), "loss_ce": mean_list(all_loss_ce)}
    if modular_model_training_mode:
        loss_value_dict.update({"loss_affinity": mean_list(all_loss_affinity),
                                "loss_dispersion": mean_list(all_loss_dispersion),
                                "loss_compactness": mean_list(all_loss_compactness)})

    return train_acc, loss_value_dict


def build_related_dirs(args, run_version="", skip_if_checkpoint_exists=True):
    base_model_checkpoint_dir = os.path.join(ModelConfig.model_checkpoint_dir, f"{args.model}_{args.dataset}")
    model_checkpoint_name = f"model" \
                            f"__bs{args.batch_size}" \
                            f"__ep{args.n_epochs}" \
                            f"__lr{args.learning_rate}"
    if "wf_affinity" in args.__dict__:
        model_checkpoint_name += f"__aff{args.wf_affinity}_dis{args.wf_dispersion}_comp{args.wf_compactness}"

    model_checkpoint_dir = os.path.join(base_model_checkpoint_dir, model_checkpoint_name)
    if skip_if_checkpoint_exists:
        checkpoint_exists = os.path.exists(model_checkpoint_dir)
        if checkpoint_exists:
            return None, None, None
    versioned_model_checkpoint_dir = os.path.join(model_checkpoint_dir, run_version)

    tensorboard_logdir = versioned_model_checkpoint_dir.replace(ModelConfig.model_checkpoint_dir,
                                                                ModelConfig.tensorboard_logdir)
    runtime_logdir = versioned_model_checkpoint_dir.replace(ModelConfig.model_checkpoint_dir,
                                                            ModelConfig.runtime_logdir)

    for curr_dir in [versioned_model_checkpoint_dir, tensorboard_logdir, runtime_logdir]:
        os.makedirs(curr_dir, exist_ok=True)

    return versioned_model_checkpoint_dir, tensorboard_logdir, runtime_logdir


def main():
    args = get_args()
    run_version = f"v{get_current_timestamp()}"

    model_checkpoint_dir, tensorboard_logdir, runtime_logdir = build_related_dirs(args, run_version,
                                                                                  skip_if_checkpoint_exists=True)
    if not model_checkpoint_dir:
        print("[Error] Model checkpoint already exists"
              "\n-> Please remove existing checkpoint or change argument: skip_if_checkpoint_exists=False")
        return

    print(f"\n[Training process started]\n"
          f"----\nmodel_checkpoint_dir: {model_checkpoint_dir}\n"
          f"----\ntensorboard_logdir: {tensorboard_logdir}\n"
          f"----\nruntime_logdir: {runtime_logdir}\n"
          f"----\n")

    tensorboard_log_writer = SummaryWriter(tensorboard_logdir, filename_suffix=f".{run_version}")
    runtime_log_writer = redirect_stdout_to_file(runtime_logdir, f"{Path(__file__).stem}.{run_version}.log")

    num_classes, train_loader, test_loader = load_dataset(dataset_type=args.dataset, batch_size=args.batch_size,
                                                          num_workers=args.dataloader_num_workers)
    model = create_modular_model(model_type=args.model, num_classes=num_classes,
                                 modular_training_mode=True)

    print(args.__dict__)
    print_model_summary(model)
    print(f"[Dataset {args.dataset}]"
          f" Train Dim {train_loader.dataset.data.shape} "
          f"- Test Dim {test_loader.dataset.data.shape}")

    modular_params = {arg: value for arg, value in args.__dict__.items() if arg.startswith("wf_")}
    model = train(model=model,
                  train_loader=train_loader, test_loader=test_loader,
                  modular_params=modular_params, learning_rate=args.learning_rate,
                  num_epochs=args.n_epochs, checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
                  checkpoint_dir=model_checkpoint_dir, tensorboard_writer=tensorboard_log_writer)

    tensorboard_log_writer.close()
    runtime_log_writer.close()


if __name__ == '__main__':
    main()
