import numpy as np
import torch
import torchvision.datasets as datasets
from torch.utils.data import DataLoader, Dataset, Subset
import pytorch_lightning as pl
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from augmentations import build_transforms

# Number of classes in each dataset. EncoderClassifier needs it for its head.
NUM_CLASSES = {"TinyImageNet": 200, "CIFAR100": 100}


class TinyImageNetSplit(Dataset):
    """One Hugging Face split of zh-plus/tiny-imagenet, read as (image, label) with a
    transform, the same interface as the torchvision datasets (CIFAR100 below).

    A few Tiny-ImageNet images are grayscale. convert("RGB") copies the one
    channel three times, so every image has 3 channels.
    """

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
        val:   n_val_per_class training images per class (Tiny-ImageNet: 50 x 200 = 10,000),
               for all tuning and model selection
        test:  the official labeled val set (Tiny-ImageNet "valid", 10,000),
               only for the final numbers

    label_fraction keeps a class-balanced part of train (1.0, 0.1 or 0.01) for
    the linear probe, fine-tuning and the supervised baseline. Pretraining uses
    1.0 and ignores the labels.

    As in the course, the training images are opened twice, once with the train
    transform and once with the eval transform, and torch's Subset picks the
    images of each split. The class-balanced splits come from scikit-learn's
    train_test_split with stratify=labels, with one fixed seed, so every model
    gets the same images.
    """

    def __init__(self, data_dir, batch_size, num_workers, dataset_name="TinyImageNet",
                 label_fraction=1.0, n_val_per_class=50, image_size=64, seed=42):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset_name = dataset_name  # "TinyImageNet" or "CIFAR100"
        self.label_fraction = label_fraction  # 1.0 = every training image
        self.n_val_per_class = n_val_per_class
        self.image_size = image_size
        self.seed = seed
        self.num_classes = NUM_CLASSES[dataset_name]

    def _load(self, train_transform, eval_transform):
        """The training images with each transform, the test set, and the train labels.

        The files are already on disk (prepare_data), so this only opens them.
        """
        if self.dataset_name == "TinyImageNet":
            ds = load_dataset("zh-plus/tiny-imagenet", cache_dir=self.data_dir)
            train = TinyImageNetSplit(ds["train"], train_transform)  # 100,000 images
            train_eval = TinyImageNetSplit(ds["train"], eval_transform)
            test = TinyImageNetSplit(ds["valid"], eval_transform)  # 10,000 images, the official val set
            # Read the label column straight from the Arrow table: fast, and no image is decoded.
            labels = np.array(ds["train"].data.column("label").to_pylist())
        else:
            train = datasets.CIFAR100(self.data_dir, train=True, transform=train_transform)  # 50,000 images, 32x32
            train_eval = datasets.CIFAR100(self.data_dir, train=True, transform=eval_transform)
            test = datasets.CIFAR100(self.data_dir, train=False, transform=eval_transform)  # 10,000 images
            labels = np.array(train.targets)
        return train, train_eval, test, labels

    def prepare_data(self):
        # Download once. Lightning calls this before setup, on one process only.
        # Only the download: the splits are built in setup.
        if self.dataset_name == "TinyImageNet":
            load_dataset("zh-plus/tiny-imagenet", cache_dir=self.data_dir)
        else:
            datasets.CIFAR100(self.data_dir, train=True, download=True)
            datasets.CIFAR100(self.data_dir, train=False, download=True)

    def setup(self, stage):
        # train.py calls fit, validate and test, and each calls setup. The splits
        # are the same every time, so they are built only on the first call.
        if hasattr(self, "train_ds"):
            return
        train_transform, eval_transform = build_transforms(self.image_size)
        train_base, train_eval_base, test_base, labels = self._load(train_transform, eval_transform)

        # Hold out n_val_per_class images of every class as val (stratify = the same share of every class).
        all_idx = np.arange(len(labels))
        train_idx, val_idx = train_test_split(
            all_idx, test_size=self.n_val_per_class * self.num_classes, stratify=labels, random_state=self.seed
        )
        # The labeled part of train: the same share of every class, the same images for every model.
        labeled_idx = train_idx
        if self.label_fraction < 1.0:
            labeled_idx, _ = train_test_split(
                train_idx, train_size=self.label_fraction, stratify=labels[train_idx], random_state=self.seed
            )

        self.labels = labels
        self.train_idx = train_idx
        self.train_ds = Subset(train_base, labeled_idx)
        # The k-NN memory bank: ALL the training images, whatever label_fraction is,
        # with the eval transform (no flip), so every model is scored with the same bank.
        self.bank_ds = Subset(train_eval_base, train_idx)
        self.val_ds = Subset(train_eval_base, val_idx)
        self.test_ds = test_base

    def _dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=self.num_workers > 0,  # keep the workers between epochs instead of starting new ones
            pin_memory=torch.cuda.is_available(),  # faster copy to the GPU; does nothing on a CPU
        )

    def train_dataloader(self):
        return self._dataloader(self.train_ds, shuffle=True)

    def bank_dataloader(self, max_size=None):
        """All training images, no flip, fixed order: the k-NN memory bank.

        max_size keeps a class-balanced part of it (None = all), with the same
        train_test_split: 5,000 of 90,000 -> 25 per class.
        """
        dataset = self.bank_ds
        if max_size is not None and max_size < len(dataset):
            keep, _ = train_test_split(
                self.train_idx, train_size=max_size, stratify=self.labels[self.train_idx], random_state=self.seed
            )
            dataset = Subset(self.bank_ds.dataset, keep)
        return self._dataloader(dataset, shuffle=False)

    def val_dataloader(self):
        return self._dataloader(self.val_ds, shuffle=False)

    def test_dataloader(self):
        return self._dataloader(self.test_ds, shuffle=False)
