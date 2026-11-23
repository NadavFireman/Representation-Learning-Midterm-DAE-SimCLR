import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from augmentations import build_transforms

NUM_CLASSES = {"TinyImageNet": 200, "CIFAR100": 100}


class TinyImageNetSplit(Dataset):
    """One Hugging Face split of zh-plus/tiny-imagenet as (image, label), like a torchvision
    dataset. convert("RGB"): a few images are grayscale."""

    def __init__(self, hf_split, transform):
        self.hf_split = hf_split
        self.transform = transform

    def __len__(self):
        return len(self.hf_split)

    def __getitem__(self, idx):
        item = self.hf_split[idx]
        return self.transform(item["image"].convert("RGB")), item["label"]  # [3, 64, 64], int


class ImageDataModule(pl.LightningDataModule):
    """Tiny-ImageNet (pretraining and evaluation) or CIFAR-100 (transfer).

    Splits, as the assignment asks:
        train: the training images minus the held-out ones (Tiny-ImageNet: 90,000)
        val:   n_val_per_class training images per class (10,000), for tuning and model selection
        test:  the official labeled val set (10,000), only for the final numbers
    label_fraction keeps a class-balanced part of train (1.0, 0.1 or 0.01).
    The splits use one fixed seed, so every model gets the same images.
    test_curve=True: val_dataloader returns two loaders, val and test, so the model also
    logs the test numbers after every epoch (testcurve_*), to plot only.
    """

    def __init__(self, data_dir, batch_size, num_workers, dataset_name="TinyImageNet",
                 label_fraction=1.0, n_val_per_class=50, image_size=64, seed=42, test_curve=False, eval_batch_size=None,
                 augment="flip"):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name  # "TinyImageNet" or "CIFAR100"
        self.label_fraction = label_fraction  # 1.0 = every training image
        self.n_val_per_class = n_val_per_class
        self.image_size = image_size
        self.seed = seed
        self.test_curve = test_curve
        self.eval_batch_size = eval_batch_size or batch_size  # val, test and the k-NN bank
        self.augment = augment  # augmentations.AUGMENTS: what the training images get
        self.num_classes = NUM_CLASSES[dataset_name]

    def _load(self, train_transform, eval_transform):
        """The training images with each transform, the test set, and the train labels."""
        if self.dataset_name == "TinyImageNet":
            ds = load_dataset("zh-plus/tiny-imagenet", cache_dir=self.data_dir)
            train = TinyImageNetSplit(ds["train"], train_transform)  # 100,000 images
            train_eval = TinyImageNetSplit(ds["train"], eval_transform)
            test = TinyImageNetSplit(ds["valid"], eval_transform)  # 10,000 images, the official val set
            labels = np.array(ds["train"]["label"])  # only the label column: no image is decoded
        else:
            train = datasets.CIFAR100(self.data_dir, train=True, transform=train_transform)  # 50,000 images, 32x32
            train_eval = datasets.CIFAR100(self.data_dir, train=True, transform=eval_transform)
            test = datasets.CIFAR100(self.data_dir, train=False, transform=eval_transform)  # 10,000 images
            labels = np.array(train.targets)
        return train, train_eval, test, labels

    def prepare_data(self):
        # Download once. The splits are built in setup.
        if self.dataset_name == "TinyImageNet":
            load_dataset("zh-plus/tiny-imagenet", cache_dir=self.data_dir)
        else:
            datasets.CIFAR100(self.data_dir, train=True, download=True)
            datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # fit, validate and test each call setup: build the splits only once.
        if hasattr(self, "train_ds"):
            return
        train_transform, eval_transform = build_transforms(self.image_size, self.augment)
        train_base, train_eval_base, test_base, labels = self._load(train_transform, eval_transform)

        # Hold out n_val_per_class images of every class as val (stratify: class-balanced).
        all_idx = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            all_idx, test_size=self.n_val_per_class * self.num_classes, stratify=labels, random_state=self.seed
        )
        # The labeled part of train, class-balanced.
        labeled_idx = train_idx
        if self.label_fraction < 1.0:
            labeled_idx, _ = train_test_split(
                train_idx, train_size=self.label_fraction, stratify=labels[train_idx], random_state=self.seed
            )

        self.labels = labels
        self.train_idx = train_idx
        self.train_ds = Subset(train_base, labeled_idx)
        # The k-NN memory bank: all the training images, with the eval transform (no flip).
        self.bank_ds = Subset(train_eval_base, train_idx)
        self.val_ds = Subset(train_eval_base, val_idx)
        self.test_ds = test_base

    def class_names(self):
        """The name of every class, by label (Tiny-ImageNet: WordNet ids). Needs setup() first."""
        if self.dataset_name == "TinyImageNet":
            return self.test_ds.hf_split.features["label"].names
        return self.test_ds.classes

    def _dataloader(self, dataset, shuffle, persistent=True):
        # persistent=False: a loader used once (the k-NN's), whose workers need not stay
        return DataLoader(
            dataset,
            batch_size=self.batch_size if shuffle else self.eval_batch_size,  # shuffle: the training set
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=persistent and self.num_workers > 0,  # keep the workers between epochs
            pin_memory=torch.cuda.is_available(),  # faster copy to the GPU
        )

    def train_dataloader(self):
        return self._dataloader(self.train_ds, shuffle=True)

    def bank_dataloader(self, max_size=None):
        """The k-NN memory bank: all training images, no flip, fixed order.
        max_size keeps a class-balanced part of it (None = all)."""
        dataset = self.bank_ds
        if max_size is not None and max_size < len(dataset):
            keep, _ = train_test_split(
                self.train_idx, train_size=max_size, stratify=self.labels[self.train_idx], random_state=self.seed
            )
            dataset = Subset(self.bank_ds.dataset, keep)
        return self._dataloader(dataset, shuffle=False, persistent=False)

    def split_dataloader(self, split, persistent=True):
        """The val or test images, no flip, fixed order."""
        return self._dataloader(self.val_ds if split == "val" else self.test_ds, shuffle=False, persistent=persistent)

    def val_dataloader(self):
        # Two loaders: validation_step gets dataloader_idx 0 (val) and 1 (test).
        if self.test_curve:
            return [self.split_dataloader("val"), self.split_dataloader("test")]
        return self.split_dataloader("val")

    def test_dataloader(self):
        return self.split_dataloader("test")
