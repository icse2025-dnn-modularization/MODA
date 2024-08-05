from functools import lru_cache

import torchvision.transforms as transforms
from torchvision.datasets import MNIST

from dataset_loader.dataset_utils import create_dataset_loader

_MNIST_MEAN = [0.1307, ]
_MNIST_STD = [0.3081, ]


def _get_default_mnist_transforms():
    train_transform = test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=_MNIST_MEAN, std=_MNIST_STD)
    ])
    return train_transform, test_transform


# cache data for running large experiments faster
@lru_cache
def _load_all_mnist_dataset(dataset_dir, custom_transforms=None):
    if not custom_transforms:
        train_transform, test_transform = _get_default_mnist_transforms()
    else:
        train_transform, test_transform = custom_transforms

    train_dataset = MNIST(root=dataset_dir,
                          train=True,
                          transform=train_transform,
                          download=True)

    test_dataset = MNIST(root=dataset_dir,
                         train=False,
                         transform=test_transform)
    return train_dataset, test_dataset


def load_mnist_dataset(batch_size, dataset_dir, num_workers=2, target_classes=None, sample_size_per_class=None):
    train_dataset, test_dataset = _load_all_mnist_dataset(dataset_dir)
    return create_dataset_loader(train_dataset=train_dataset, test_dataset=test_dataset,
                                 target_classes=target_classes, sample_size_per_class=sample_size_per_class,
                                 batch_size=batch_size, num_workers=num_workers)