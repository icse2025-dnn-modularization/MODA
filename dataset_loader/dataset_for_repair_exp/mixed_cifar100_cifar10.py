from dataset_loader.cifar10 import _load_all_cifar10_dataset, _get_default_cifar10_transforms
from dataset_loader.cifar100 import _load_all_cifar100_dataset
from dataset_loader.dataset_for_repair_exp.base_mixed_dataset import MixedDataset
from dataset_loader.dataset_utils import create_dataset_loader, filter_dataset


def load_mixed_cifar100_cifar10_repair_dataset(batch_size,
                                               cifar100_dataset_dir, cifar10_dataset_dir,
                                               mixed_class=None,
                                               num_workers=2):
    """
        according to CNNSplitter: https://github.com/qibinhang/CNNSplitter/blob/main/src/experiments/patch/apply_patch.py
        select following CIFAR100 labels for repair experiements:
            {0: apple, 2: baby, 5: bed, 8: bicycle, 9: bottle, 12: bridge, 15: camel, 22: clock, 70: rose} from CIFAR-100
    """

    # to compare with strong models trained on CIFAR10, the mixing dataset will be use CIFAR10 dataset transform
    _, cifar10_test_transform = _get_default_cifar10_transforms()

    # CIFAR100
    selected_cifar100_classes = [0, 2, 5, 8, 9, 12, 15, 22, 70]

    # skip default augmentation in the training set to be able to train weak models
    cifar100_train_dataset, cifar100_test_dataset = _load_all_cifar100_dataset(cifar100_dataset_dir,
                                                                               custom_transforms=(
                                                                                   cifar10_test_transform,
                                                                                   cifar10_test_transform))

    cifar100_num_classes, cifar100_train_dataset = filter_dataset(cifar100_train_dataset,
                                                                  target_classes=selected_cifar100_classes,
                                                                  sample_size_per_class=None, transform_label=True)
    _, cifar100_test_dataset = filter_dataset(cifar100_test_dataset,
                                              target_classes=selected_cifar100_classes,
                                              sample_size_per_class=None, transform_label=True)

    # CIFAR10
    cifar10_mixed_class = mixed_class
    cifar10_train_dataset, cifar10_test_dataset = _load_all_cifar10_dataset(cifar10_dataset_dir)
    _, cifar10_train_dataset = filter_dataset(cifar10_train_dataset,
                                              target_classes=[cifar10_mixed_class],
                                              sample_size_per_class=500, transform_label=True)
    _, cifar10_test_dataset = filter_dataset(cifar10_test_dataset,
                                             target_classes=[cifar10_mixed_class],
                                             sample_size_per_class=100, transform_label=True)

    mixed_train_dataset = MixedDataset(cifar100_train_dataset, cifar10_train_dataset, shuffle=True)
    mixed_test_dataset = MixedDataset(cifar100_test_dataset, cifar10_test_dataset)

    return create_dataset_loader(train_dataset=mixed_train_dataset, test_dataset=mixed_test_dataset,
                                 target_classes=None, sample_size_per_class=None,
                                 batch_size=batch_size, num_workers=num_workers)
