from functools import lru_cache

import numpy as np
import torchvision.transforms as transforms
from torchvision.datasets import SVHN as BaseSVHN

from dataset_loader.dataset_utils import create_dataset_loader

_SVHN_MEAN = [0.485, 0.456, 0.406]
_SVHN_STD = [0.229, 0.224, 0.225]


class SVHN(BaseSVHN):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.classes = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

    @property
    def targets(self):
        return self.labels

    @targets.setter
    def targets(self, new_targets):
        self.labels = new_targets


def _get_default_svhn_transforms():
    normalize = transforms.Normalize(mean=_SVHN_MEAN, std=_SVHN_STD)
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
def _load_all_svhn_dataset(dataset_dir, custom_transforms=None):
    if not custom_transforms:
        train_transform, test_transform = _get_default_svhn_transforms()
    else:
        train_transform, test_transform = custom_transforms

    train_dataset = SVHN(root=dataset_dir,
                         split="train",
                         transform=train_transform,
                         download=True)

    test_dataset = SVHN(root=dataset_dir,
                        split="test",
                        transform=test_transform,
                        download=True)

    return train_dataset, test_dataset


def load_svhn_dataset(batch_size, dataset_dir, num_workers=2, target_classes=None, sample_size_per_class=None):
    train_dataset, test_dataset = _load_all_svhn_dataset(dataset_dir)
    return create_dataset_loader(train_dataset=train_dataset, test_dataset=test_dataset,
                                 target_classes=target_classes, sample_size_per_class=sample_size_per_class,
                                 batch_size=batch_size, num_workers=num_workers)
