import torch
from torchvision.transforms import v2


def build_transforms(image_size=64):
    """Train and eval transforms for RGB images (Tiny-ImageNet and the transfer set).

    Pixels stay in [0, 1], with no Normalize. The autoencoder's decoder ends in a
    Sigmoid, so its output is in (0, 1) and the clean target must be in the same
    range. Every mode uses these transforms, so the encoder always sees the same
    kind of input in pretraining, k-NN, the linear probe and fine-tuning.

    Every image leaves as image_size x image_size, whatever size it came in.
    An image that is already 64x64 (Tiny-ImageNet) passes through unchanged.
    Any other image keeps its proportions: Resize scales the shorter side to 64,
    and CenterCrop cuts the longer side to 64. Nothing is stretched. CIFAR-100
    (32x32, square) is only upsampled.

    Train adds one augmentation, a horizontal flip: a mirrored photo shows the
    same object, so it cannot change the class. The noise of the denoising
    autoencoder is not added here. The model adds it, because its loss needs
    the clean image too.
    """
    train_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
            v2.Resize(image_size, antialias=True),  # shorter side -> 64, proportions kept
            v2.CenterCrop(image_size),  # longer side cut to 64 -> [3, 64, 64]
            v2.RandomHorizontalFlip(p=0.5),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
            v2.Resize(image_size, antialias=True),  # shorter side -> 64, proportions kept
            v2.CenterCrop(image_size),  # longer side cut to 64 -> [3, 64, 64]
        ]
    )
    return train_transform, eval_transform
