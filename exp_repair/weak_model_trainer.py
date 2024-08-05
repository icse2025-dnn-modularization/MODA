import __config_pythonpath
import argparse
from collections import defaultdict
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter
from dataset_loader import supported_repair_datasets, load_repair_dataset
from model_trainer import build_related_dirs, train
from exp_repair.repair_models import create_weak_model, supported_weak_models
from models.model_utils import print_model_summary, get_runtime_device, redirect_stdout_to_file

DEVICE = get_runtime_device()


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=supported_weak_models, required=True)
    parser.add_argument('--dataset', type=str, choices=supported_repair_datasets, required=True)

    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--dataloader_num_workers', type=int, default=2)

    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument('--checkpoint_every_n_epochs', type=int, default=-1)

    args = parser.parse_args()

    return args


def main():
    args = get_args()
    # run_version = f"v{get_current_timestamp()}"
    # run_version = f"v_testRepair_augmented"
    # run_version = f"v_testRepair_mod"
    # run_version = f"v_testRepair_std"
    run_version = f"v_testRepair_weak_mod"

    model_checkpoint_dir, tensorboard_logdir, runtime_logdir = build_related_dirs(args, run_version,
                                                                                  skip_if_checkpoint_exists=False)
    iter_num_classes = 5
    for mixed_class in range(iter_num_classes):
        run_version_mc = f"{run_version}.mc{mixed_class}"
        tensorboard_log_writer = SummaryWriter(tensorboard_logdir, filename_suffix=f".{run_version_mc}")
        runtime_log_writer = redirect_stdout_to_file(runtime_logdir, f"{Path(__file__).stem}.{run_version_mc}.log")
        num_classes, train_loader, test_loader = load_repair_dataset(for_model="weak",
                                                                     dataset_type=args.dataset,
                                                                     batch_size=args.batch_size,
                                                                     num_workers=args.dataloader_num_workers,
                                                                     mixed_class=mixed_class)
        model = create_weak_model(model_type=args.model, num_classes=num_classes, modular_training_mode=True)

        print(args.__dict__)
        print_model_summary(model)
        print(f"[Dataset {args.dataset}]"
              f" Train Dim {train_loader.dataset.data.shape} "
              f"- Test Dim {test_loader.dataset.data.shape} "
              f"- Mixed Class {mixed_class}")

        # modular_params = defaultdict(float)
        # modular_params = {"wf_affinity": 0.5, "wf_dispersion": 0.5, "wf_compactness": 0.5}
        modular_params = {"wf_affinity": 1.0, "wf_dispersion": 1.0, "wf_compactness": 0.3}
        model = train(model=model,
                      train_loader=train_loader, test_loader=test_loader,
                      modular_params=modular_params, learning_rate=args.learning_rate,
                      num_epochs=args.n_epochs, checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
                      checkpoint_dir=model_checkpoint_dir, checkpoint_model_name=f"weak_model.mc{mixed_class}",
                      tensorboard_writer=tensorboard_log_writer)
        tensorboard_log_writer.close()
        runtime_log_writer.close()


if __name__ == '__main__':
    main()
