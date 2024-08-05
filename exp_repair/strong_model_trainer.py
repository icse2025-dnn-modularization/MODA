import __config_pythonpath
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from dataset_loader import load_repair_dataset
from model_trainer import build_related_dirs, train
from exp_repair.weak_model_trainer import get_args
from models.model_utils import print_model_summary, get_runtime_device, redirect_stdout_to_file
from models import create_modular_model

DEVICE = get_runtime_device()


def main():
    args = get_args()
    run_version = f"v_testRepair_strong_mod"

    model_checkpoint_dir, tensorboard_logdir, runtime_logdir = build_related_dirs(args, run_version,
                                                                                  skip_if_checkpoint_exists=False)
    tensorboard_log_writer = SummaryWriter(tensorboard_logdir, filename_suffix=f".{run_version}")
    runtime_log_writer = redirect_stdout_to_file(runtime_logdir, f"{Path(__file__).stem}.{run_version}.log")
    num_classes, train_loader, test_loader = load_repair_dataset(for_model="strong",
                                                                 dataset_type=args.dataset,
                                                                 batch_size=args.batch_size,
                                                                 num_workers=args.dataloader_num_workers)
    model = create_modular_model(model_type=args.model, num_classes=num_classes, modular_training_mode=True)

    print(args.__dict__)
    print_model_summary(model)
    print(f"[Dataset {args.dataset}]"
          f" Train Dim {train_loader.dataset.data.shape} "
          f"- Test Dim {test_loader.dataset.data.shape} ")

    # modular_params = {"wf_affinity": 0.5, "wf_dispersion": 0.5, "wf_compactness": 0.5}
    modular_params = {"wf_affinity": 1.0, "wf_dispersion": 1.0, "wf_compactness": 0.3}
    model = train(model=model,
                  train_loader=train_loader, test_loader=test_loader,
                  modular_params=modular_params, learning_rate=args.learning_rate,
                  num_epochs=args.n_epochs, checkpoint_every_n_epochs=args.checkpoint_every_n_epochs,
                  checkpoint_dir=model_checkpoint_dir, checkpoint_model_name=f"strong_model",
                  tensorboard_writer=tensorboard_log_writer)

    tensorboard_log_writer.close()
    runtime_log_writer.close()


if __name__ == '__main__':
    main()
