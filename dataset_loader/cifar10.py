from functools import lru_cache

import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10

from dataset_loader.dataset_utils import create_dataset_loader

_CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
_CIFAR10_STD = [0.2471, 0.2435, 0.2616]


def _get_default_cifar10_transforms():
    normalize = transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STD)
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
def _load_all_cifar10_dataset(dataset_dir, custom_transforms=None):
    if not custom_transforms:
        train_transform, test_transform = _get_default_cifar10_transforms()
    else:
        train_transform, test_transform = custom_transforms

    train_dataset = CIFAR10(root=dataset_dir,
                            train=True,
                            transform=train_transform,
                            download=True)

    test_dataset = CIFAR10(root=dataset_dir,
                           train=False,
                           transform=test_transform)

    return train_dataset, test_dataset


def load_cifar10_dataset(batch_size, dataset_dir, num_workers=2, target_classes=None, sample_size_per_class=None):
    train_dataset, test_dataset = _load_all_cifar10_dataset(dataset_dir)
    return create_dataset_loader(train_dataset=train_dataset, test_dataset=test_dataset,
                                 target_classes=target_classes, sample_size_per_class=sample_size_per_class,
                                 batch_size=batch_size, num_workers=num_workers)
