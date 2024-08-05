import copy

import numpy as np
from torch import Tensor

from torch.utils.data import DataLoader


def create_dataset_loader(train_dataset, test_dataset, target_classes, sample_size_per_class,
                          shuffle=True, batch_size=128, num_workers=2):
    num_classes = len(train_dataset.classes)

    do_filter = target_classes or sample_size_per_class
    if do_filter:
        num_classes, train_dataset = filter_dataset(train_dataset, target_classes, sample_size_per_class)
        _, test_dataset = filter_dataset(test_dataset, target_classes, sample_size_per_class)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=shuffle,
                              num_workers=num_workers,
                              pin_memory=True)

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=num_workers,
                             pin_memory=True)

    return num_classes, train_loader, test_loader


def filter_dataset(dataset, target_classes, sample_size_per_class, transform_label=False):
    if target_classes is None:
        target_classes = dataset.classes
    target_classes = sorted(target_classes)

    if isinstance(dataset.targets, list):
        targets = np.squeeze(dataset.targets)
    elif isinstance(dataset.targets, Tensor):
        targets = np.squeeze(dataset.targets)
    elif isinstance(dataset.targets, np.ndarray):
        targets = dataset.targets
    else:
        raise NotImplementedError()

    # filter by target_classes
    filtered_indices = np.isin(targets, target_classes).nonzero()[0]

    # filter by sample_size_per_class
    if sample_size_per_class:
        filtered_indices = _sample_class_wise(all_sample_labels=targets, considering_sample_indices=filtered_indices,
                                              considering_classes=target_classes,
                                              sample_size_per_class=sample_size_per_class)

    # get new data after being filtered
    filtered_data = dataset.data[filtered_indices]
    filtered_targets = targets[filtered_indices]
    assert np.all(np.unique(filtered_targets) == target_classes)

    filtered_dataset = copy.copy(dataset)
    filtered_dataset.data = filtered_data
    filtered_dataset.classes = [dataset.classes[c] for c in target_classes]
    if transform_label:
        # transforming classes according to data after filtered (e.g, before [1,8,9] -> after: [0,1,2])
        filtered_dataset.targets = np.vectorize(lambda x: target_classes.index(x))(filtered_targets).tolist()
        filtered_dataset.class_to_idx = {_class: i for i, _class in enumerate(filtered_dataset.classes)}
    else:
        filtered_dataset.targets = filtered_targets.tolist()
        filtered_dataset.class_to_idx = {key: dataset.class_to_idx[key] for key in filtered_dataset.classes}

    return len(target_classes), filtered_dataset


def _sample_class_wise(all_sample_labels, considering_sample_indices, considering_classes, sample_size_per_class):
    considering_sample_labels = all_sample_labels[considering_sample_indices]
    if isinstance(sample_size_per_class, int):
        class_wise_sample_sizes = {label: sample_size_per_class for label in considering_classes}
    elif isinstance(sample_size_per_class, float):
        assert 0 < sample_size_per_class < 1, "proportion sample size must be 0 < s < 1"
        class_wise_sample_sizes = {
            label: int(np.sum(considering_sample_labels == label) * sample_size_per_class)
            for label in considering_classes}
    else:
        raise NotImplementedError()

    filtered_indices = np.concatenate(
        [np.random.choice(considering_sample_indices[considering_sample_labels == label],
                          class_wise_sample_sizes[label],
                          replace=False)
         for label in considering_classes])
    return filtered_indices
