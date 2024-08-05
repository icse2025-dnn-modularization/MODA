import logging
import os

from models.model_utils import set_global_seed

seed = 42
set_global_seed(seed)


class BaseConfig:
    project_dir = os.path.dirname(os.path.realpath(__file__))
    resource_dir = os.path.join(project_dir, "raw_data")


class DatasetConfig:
    dataset_dir = os.path.join(BaseConfig.resource_dir, "torchvision_datasets")


class ModelConfig:
    model_base_dir = os.path.join(BaseConfig.resource_dir, "models")

    model_checkpoint_dir = os.path.join(model_base_dir, "model_checkpoints")
    module_checkpoint_dir = os.path.join(model_base_dir, "module_checkpoints")

    runtime_logdir = os.path.join(model_base_dir, "runtime_log")
    tensorboard_logdir = os.path.join(model_base_dir, "tensorboard_log")


logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [%(module)s] %(message)s",
                    datefmt='%Y-%m-%d %H:%M:%S')
