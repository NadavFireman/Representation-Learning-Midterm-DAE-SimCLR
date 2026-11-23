import torch
from torchvision.transforms import v2

# Per-channel mean and std of the Tiny-ImageNet train split (90,000 images), pixels in [0, 1]
TINY_IMAGENET_MEAN = (0.4804, 0.4482, 0.3976)
TINY_IMAGENET_STD = (0.2764, 0.2688, 0.2816)

# --augment: what the training images of a classifier get
AUGMENTS = {
    "none": [],
    "flip": [v2.RandomHorizontalFlip(p=0.5)],
    "crop": [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(64, padding=4)],  # a shift of up to 4 pixels
    "crop_rotate": [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(64, padding=4), v2.RandomRotation(10)],
    "crop_jitter": [v2.RandomHorizontalFlip(p=0.5), v2.RandomCrop(64, padding=4),
                    v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)],
    "flip_rotate": [v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(10)],
    "flip_jitter": [v2.RandomHorizontalFlip(p=0.5), v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)],
    "flip_rotate_jitter": [v2.RandomHorizontalFlip(p=0.5), v2.RandomRotation(10),
                           v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)],
}


def build_transforms(image_size=64, augment="flip"):
    """Train and eval transforms. Every image leaves as image_size x image_size, normalized
    with the Tiny-ImageNet values (the transfer set too). Train adds AUGMENTS[augment]
    (the DAE: a horizontal flip). The DAE noise is added in the model, because its loss
    needs the clean image."""
    resize = [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),  # uint8 [0, 255] -> float [0, 1]
        v2.Resize(image_size, antialias=True),  # shorter side -> 64, proportions kept
        v2.CenterCrop(image_size),  # longer side cut to 64 -> [3, 64, 64]
    ]
    normalize = v2.Normalize(TINY_IMAGENET_MEAN, TINY_IMAGENET_STD)
    train_transform = v2.Compose(resize + AUGMENTS[augment] + [normalize])
    eval_transform = v2.Compose(resize + [normalize])
    return train_transform, eval_transform
