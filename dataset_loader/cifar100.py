from functools import lru_cache

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100

from configs import DatasetConfig
from dataset_loader.dataset_utils import create_dataset_loader

_CIFAR100_MEAN = [0.5071, 0.4867, 0.4408]
_CIFAR100_STD = [0.2675, 0.2565, 0.2761]


def _get_default_cifar100_transforms():
    normalize = transforms.Normalize(mean=_CIFAR100_MEAN, std=_CIFAR100_STD)
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, 4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize
    ])
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])
    return train_transform, test_transform


# cache data for running large experiments faster
@lru_cache
def _load_all_cifar100_dataset(dataset_dir, custom_transforms=None):
    if not custom_transforms:
        train_transform, test_transform = _get_default_cifar100_transforms()
    else:
        train_transform, test_transform = custom_transforms

    train_dataset = CIFAR100(root=dataset_dir,
                             train=True,
                             transform=train_transform,
                             download=True)

    test_dataset = CIFAR100(root=dataset_dir,
                            train=False,
                            transform=test_transform)

    return train_dataset, test_dataset


def load_cifar100_dataset(batch_size, dataset_dir, num_workers=2, target_classes=None, sample_size_per_class=None):
    train_dataset, test_dataset = _load_all_cifar100_dataset(dataset_dir)
    return create_dataset_loader(train_dataset=train_dataset, test_dataset=test_dataset,
                                 target_classes=target_classes, sample_size_per_class=sample_size_per_class,
                                 batch_size=batch_size, num_workers=num_workers)
