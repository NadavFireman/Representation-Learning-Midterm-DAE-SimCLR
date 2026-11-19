import argparse
import os

import config
from datamodule import NUM_CLASSES
from model import DenoisingAutoEncoder, EncoderClassifier, SupervisedEncoder

# One name from this tuple. config.MODEL_NAME is the default.
# pretrain: DenoisingAutoEncoder (--noise_level 0 = plain autoencoder, the AE baseline).
# pretrain_supervise_encoder: SupervisedEncoder, the same encoder trained with labels.
# linear_probe and finetune: EncoderClassifier, which loads a saved encoder.
# knn: no training, k-NN accuracy of a saved encoder's h.
MODES = (
    "pretrain",
    "pretrain_supervise_encoder",
    "linear_probe",
    "finetune",
    "knn",
)
DATASETS = {"tinyimagenet": "TinyImageNet", "cifar100": "CIFAR100"}
# Which encoder linear_probe, finetune and knn use:
# dae: the one pretrain saved with the same --noise_type / --noise_level / --latent_dim / --seed.
# supervised: the one pretrain_supervise_encoder saved with the same --latent_dim / --label_fraction / --seed.
# random: no file, a random-init encoder (baseline: what the architecture gives for free).
# --pretrain_dataset says where that encoder was trained (default Tiny-ImageNet), so
# "--dataset cifar100" evaluates a Tiny-ImageNet encoder on CIFAR-100 (transfer).
ENCODERS = ("dae", "supervised", "random")


def encoder_path(encoder, pretrain_dataset):
    """File name of a saved encoder, built from everything it was trained with.

    The course used one file per dataset, and every pretrain run overwrote it.
    Here every encoder keeps its own file: the dataset it was trained on, its
    hyperparameters and its training seed are all in the name, so a CIFAR-100
    baseline or a second seed never overwrites a Tiny-ImageNet encoder.
    A search trial (--search, proxy setting) gets "_search" in the name, like its
    log folder in train.py, so it never overwrites the full run of the same
    hyperparameters.
    """
    if encoder == "random":
        return None
    if encoder == "supervised":
        name = f"supervised_{pretrain_dataset}_{config.LATENT_DIM}_{config.LABEL_FRACTION}_s{config.TRAIN_SEED}"
    else:
        name = f"dae_{pretrain_dataset}_{config.NOISE_TYPE}_{config.NOISE_LEVEL}_{config.LATENT_DIM}_s{config.TRAIN_SEED}"
    if config.SEARCH:
        name += "_search"
    return os.path.join(config.OUTPUT_DIR, "encoders", name + ".pt")


def parse_args():
    parser = argparse.ArgumentParser(description="Train one model of the midterm.")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default=config.MODEL_NAME,
        help="One model name. Default: config.MODEL_NAME.",
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASETS),
        default="tinyimagenet",
        help="tinyimagenet, or cifar100 for transfer.",
    )
    # The DAE hyperparameters of the search. Default: the value in config.
    parser.add_argument("--noise_type", choices=("gaussian", "mask", "salt_pepper"), default=config.NOISE_TYPE)
    parser.add_argument("--noise_level", type=float, default=config.NOISE_LEVEL)
    parser.add_argument("--latent_dim", type=int, default=config.LATENT_DIM)
    parser.add_argument("--label_fraction", type=float, default=config.LABEL_FRACTION, help="1.0, 0.1 or 0.01")
    parser.add_argument(
        "--encoder",
        choices=ENCODERS,
        default="dae",
        help="The encoder linear_probe, finetune and knn load.",
    )
    parser.add_argument(
        "--pretrain_dataset",
        choices=tuple(DATASETS),
        default="tinyimagenet",
        help="Where the loaded encoder was trained. Not used by the pretrain modes.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=config.TRAIN_SEED,
        help="Training seed (initialization, data order, noise). The splits and label subsets keep config.SEED.",
    )
    parser.add_argument(
        "--search",
        action="store_true",
        help="One trial of the hyperparameter search: proxy setting, no test, one row in sweep_results.csv.",
    )
    parser.add_argument(
        "--output_dir",
        default=config.OUTPUT_DIR,
        help="Where logs, checkpoints, encoders and sweep_results.csv go. On Colab: a Google Drive folder.",
    )
    args = parser.parse_args()
    config.OUTPUT_DIR = args.output_dir
    config.DATASET = DATASETS[args.dataset]
    config.NOISE_TYPE = args.noise_type
    config.NOISE_LEVEL = args.noise_level
    config.LATENT_DIM = args.latent_dim
    config.LABEL_FRACTION = args.label_fraction
    config.TRAIN_SEED = args.seed
    config.SEARCH = args.search
    # pretrain writes the encoder (named after the dataset it trains on),
    # the other modes read one (named after --pretrain_dataset).
    if args.mode == "pretrain":
        config.ENCODER_PATH = encoder_path("dae", config.DATASET)
    elif args.mode == "pretrain_supervise_encoder":
        config.ENCODER_PATH = encoder_path("supervised", config.DATASET)
    else:
        config.ENCODER_PATH = encoder_path(args.encoder, DATASETS[args.pretrain_dataset])
    return args


def build_model(mode):
    num_classes = NUM_CLASSES[config.DATASET]

    if mode == "pretrain":
        return DenoisingAutoEncoder(
            learning_rate=config.LEARNING_RATE,
            latent_dim=config.LATENT_DIM,
            noise_type=config.NOISE_TYPE,
            noise_level=config.NOISE_LEVEL,
        )

    if mode == "pretrain_supervise_encoder":
        return SupervisedEncoder(
            learning_rate=config.LEARNING_RATE,
            num_classes=num_classes,
            latent_dim=config.LATENT_DIM,
        )

    # linear_probe, finetune and knn: the saved encoder (or a random one) + a new head.
    # knn only uses model.encoder, frozen.
    return EncoderClassifier(
        learning_rate=config.FINETUNE_LEARNING_RATE if mode == "finetune" else config.LEARNING_RATE,
        num_classes=num_classes,
        latent_dim=config.LATENT_DIM,
        encoder_path=config.ENCODER_PATH,
        freeze_encoder=(mode != "finetune"),
    )
