from PIL import Image
import numpy as np
from torchvision.datasets import VisionDataset


class MixedDataset:
    def __init__(self, dataset_1, dataset_2, transpose_img_dim=None, shuffle=False):
        assert isinstance(dataset_1, VisionDataset) and isinstance(dataset_2, VisionDataset)

        self.data = np.concatenate([dataset_1.data, dataset_2.data], axis=0)

        assert transpose_img_dim is None or isinstance(transpose_img_dim, tuple)
        self.transpose_img_dim = transpose_img_dim

        # handle targets (labels)
        assert tuple(dataset_2.class_to_idx.values()) == tuple(
            range(len(dataset_2.class_to_idx))), "error, need to transform_label before reaching here"
        dataset_2_norm_targets = np.vectorize(lambda old_class_idx: old_class_idx + len(dataset_1.classes)) \
            (dataset_2.targets).tolist()
        self.targets = dataset_1.targets + dataset_2_norm_targets

        # always use "dataset_1" transforms
        self.transform = dataset_1.transform
        self.target_transform = dataset_1.target_transform

        # merge metadata
        self.classes = dataset_1.classes + dataset_2.classes
        assert len(set(self.classes)) == len(dataset_1.classes) + len(dataset_2.classes)
        self.class_to_idx = {_class: i for i, _class in enumerate(self.classes)}

        if shuffle:
            self._shuffle_data()

    def _shuffle_data(self):
        indices = np.arange(len(self.data))
        np.random.shuffle(indices)
        self.data = self.data[indices]
        self.targets = [self.targets[i] for i in indices]

    def __getitem__(self, index: int):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        if getattr(self, "transpose_img_dim", None):
            img = np.transpose(img, self.transpose_img_dim)
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)
