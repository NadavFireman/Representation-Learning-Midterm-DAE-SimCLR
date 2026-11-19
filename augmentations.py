import torch
from torchvision.transforms import v2

# Per-channel mean and std of the Tiny-ImageNet train split (90,000 images, val and
# test not included), on pixels in [0, 1]. Like (0.1307,), (0.3081,) for MNIST in the course.
TINY_IMAGENET_MEAN = (0.4804, 0.4482, 0.3976)
TINY_IMAGENET_STD = (0.2764, 0.2688, 0.2816)


def build_transforms(image_size=64):
    """Train and eval transforms for RGB images (Tiny-ImageNet and the transfer set).

    As in the course: ToImage, ToDtype, then Normalize at the end. The transfer set
    is normalized with the Tiny-ImageNet values too, because that is what the
    pretrained encoder learned on.

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
            v2.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )
    eval_transform = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
            v2.Resize(image_size, antialias=True),  # shorter side -> 64, proportions kept
            v2.CenterCrop(image_size),  # longer side cut to 64 -> [3, 64, 64]
            v2.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform
