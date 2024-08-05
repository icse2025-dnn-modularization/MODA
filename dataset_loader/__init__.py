import os

from configs import DatasetConfig
from dataset_loader.cifar10 import load_cifar10_dataset
from dataset_loader.cifar100 import load_cifar100_dataset
from dataset_loader.dataset_for_repair_exp.cifar10_for_repair import \
    load_weak_cifar10_for_repair_dataset, load_strong_cifar10_for_repair_dataset
from dataset_loader.dataset_for_repair_exp.svhn_for_repair import \
    load_weak_svhn_for_repair_dataset, load_strong_svhn_for_repair_dataset
# from dataset_loader.dataset_for_repair_exp.mixed_cifar100_cifar10 import \
#     load_mixed_cifar100_cifar10_repair_dataset
from dataset_loader.mnist import load_mnist_dataset
from dataset_loader.svhn import load_svhn_dataset

supported_std_datasets = ["cifar10", "cifar100", "svhn"]

supported_repair_datasets = ["mixed_cifar10_for_repair", "mixed_svhn_for_repair"]


def load_dataset(dataset_type, *args, **kwargs):
    default_dataset_dir = os.path.join(DatasetConfig.dataset_dir, dataset_type)
    if dataset_type == "mnist":
        return load_mnist_dataset(*args, dataset_dir=default_dataset_dir, **kwargs)
    elif dataset_type == "cifar10":
        return load_cifar10_dataset(*args, dataset_dir=default_dataset_dir, **kwargs)
    elif dataset_type == "cifar100":
        return load_cifar100_dataset(*args, dataset_dir=default_dataset_dir, **kwargs)
    elif dataset_type == "svhn":
        return load_svhn_dataset(*args, dataset_dir=default_dataset_dir, **kwargs)
    else:
        raise Exception(f"dataset is not supported: {dataset_type}")


def load_repair_dataset(for_model, dataset_type, *args, **kwargs):
    # if dataset_type == "mixed_cifar100_cifar10_for_repair":
    #     cifar100_dataset_dir = os.path.join(DatasetConfig.dataset_dir, "cifar100")
    #     cifar10_dataset_dir = os.path.join(DatasetConfig.dataset_dir, "cifar10")
    #     return load_mixed_cifar100_cifar10_repair_dataset(*args,
    #                                                       cifar100_dataset_dir=cifar100_dataset_dir,
    #                                                       cifar10_dataset_dir=cifar10_dataset_dir,
    #                                                       **kwargs)
    #
    if dataset_type == "mixed_cifar10_for_repair":
        cifar10_dataset_dir = os.path.join(DatasetConfig.dataset_dir, "cifar10")
        if for_model == "weak":
            return load_weak_cifar10_for_repair_dataset(*args,
                                                        dataset_dir=cifar10_dataset_dir,
                                                        **kwargs)
        elif for_model == "strong":
            return load_strong_cifar10_for_repair_dataset(*args, dataset_dir=cifar10_dataset_dir,
                                                          **kwargs)
    elif dataset_type == "mixed_svhn_for_repair":
        cifar10_dataset_dir = os.path.join(DatasetConfig.dataset_dir, "svhn")
        if for_model == "weak":
            return load_weak_svhn_for_repair_dataset(*args,
                                                     dataset_dir=cifar10_dataset_dir,
                                                     **kwargs)
        elif for_model == "strong":
            return load_strong_svhn_for_repair_dataset(*args, dataset_dir=cifar10_dataset_dir,
                                                       **kwargs)

    raise Exception(f"dataset is not supported: {dataset_type} ({for_model})")


if __name__ == '__main__':
    load_dataset("mnist", batch_size=128)
    # load_dataset("svhn", batch_size=128)
