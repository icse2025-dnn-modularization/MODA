import os
import pickle

from dataset_loader.cifar10 import _load_all_cifar10_dataset, _get_default_cifar10_transforms
from dataset_loader.dataset_for_repair_exp.base_mixed_dataset import MixedDataset
from dataset_loader.dataset_utils import create_dataset_loader, filter_dataset

"""
select following CIFAR10 labels for repair experiements:
- 5 classes for strong model: {'airplane': 0, 'automobile': 1, 'bird': 2, 'cat': 3, 'deer': 4}
- 6 classes for weak model: {'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9, 'MIXED': 10}
"""

SELECTED_CIFAR10_STRONG_CLASSES = [0, 1, 2, 3, 4]
SELECTED_CIFAR10_WEAK_CLASSES = [5, 6, 7, 8, 9]


def load_weak_cifar10_for_repair_dataset(*args, **kwargs):
    assert len(args) == 0, "this function doesn't accept positional arguments"

    # sample weak dataset only once,
    # then save to local file to avoid randomness of later sampling

    repair_dataset_dir = os.path.join(kwargs["dataset_dir"], "mod_repair_dataset")
    os.makedirs(repair_dataset_dir, exist_ok=True)
    mixed_class = kwargs["mixed_class"]
    cache_filepath = os.path.join(repair_dataset_dir, f"weak_cifar10.MC{mixed_class}.pkl")

    if os.path.exists(cache_filepath):
        print(f"[cifar10_for_repair] loading sampled dataset from local file [{cache_filepath}]")
        with open(cache_filepath, "rb") as output_file:
            data_tuple = pickle.load(output_file)
    else:
        data_tuple = _load_weak_cifar10_for_repair_dataset(**kwargs)
        with open(cache_filepath, "wb") as output_file:
            pickle.dump(data_tuple, output_file)

    return data_tuple


def _load_weak_cifar10_for_repair_dataset(batch_size,
                                          dataset_dir,
                                          mixed_class=None,
                                          num_workers=2):
    assert mixed_class in SELECTED_CIFAR10_STRONG_CLASSES, "mixed classes must be in selected (strong) classes"

    # CIFAR10 [WEAK]
    _, cifar10_test_transform = _get_default_cifar10_transforms()

    # skip default augmentation in the training set to be able to train weak models
    cifar10_train_dataset, cifar10_test_dataset = _load_all_cifar10_dataset(dataset_dir,
                                                                            custom_transforms=(
                                                                                cifar10_test_transform,
                                                                                cifar10_test_transform))

    sample_size_per_class = 0.1
    _, weak_cifar10_train_dataset = filter_dataset(cifar10_train_dataset,
                                                   target_classes=SELECTED_CIFAR10_WEAK_CLASSES,
                                                   sample_size_per_class=sample_size_per_class,
                                                   transform_label=True)
    _, weak_cifar10_test_dataset = filter_dataset(cifar10_test_dataset,
                                                  target_classes=SELECTED_CIFAR10_WEAK_CLASSES,
                                                  sample_size_per_class=None, transform_label=True)
    # CIFAR10 [MIXED]
    _, mixed_cifar10_train_dataset = filter_dataset(cifar10_train_dataset,
                                                    target_classes=[mixed_class],
                                                    sample_size_per_class=sample_size_per_class,
                                                    transform_label=True)
    _, mixed_cifar10_test_dataset = filter_dataset(cifar10_test_dataset,
                                                   target_classes=[mixed_class],
                                                   sample_size_per_class=None, transform_label=True)

    mixed_train_dataset = MixedDataset(weak_cifar10_train_dataset, mixed_cifar10_train_dataset, shuffle=True)
    mixed_test_dataset = MixedDataset(weak_cifar10_test_dataset, mixed_cifar10_test_dataset, shuffle=False)

    return create_dataset_loader(train_dataset=mixed_train_dataset, test_dataset=mixed_test_dataset,
                                 target_classes=None, sample_size_per_class=None,
                                 shuffle=False, batch_size=batch_size, num_workers=num_workers)


def load_strong_cifar10_for_repair_dataset(batch_size,
                                           dataset_dir,
                                           num_workers=2):
    # CIFAR10 [STRONG]
    cifar10_train_dataset, cifar10_test_dataset = _load_all_cifar10_dataset(dataset_dir)
    _, strong_cifar10_train_dataset = filter_dataset(cifar10_train_dataset,
                                                     target_classes=SELECTED_CIFAR10_STRONG_CLASSES,
                                                     sample_size_per_class=None,
                                                     transform_label=True)
    _, strong_cifar10_test_dataset = filter_dataset(cifar10_test_dataset,
                                                    target_classes=SELECTED_CIFAR10_STRONG_CLASSES,
                                                    sample_size_per_class=None, transform_label=True)

    return create_dataset_loader(train_dataset=strong_cifar10_train_dataset, test_dataset=strong_cifar10_test_dataset,
                                 target_classes=None, sample_size_per_class=None,
                                 batch_size=batch_size, num_workers=num_workers)
