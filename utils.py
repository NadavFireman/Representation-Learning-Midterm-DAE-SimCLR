import argparse
import os

import config
import run_info
from augmentations import AUGMENTS
from datamodule import NUM_CLASSES
from model import (ACTIVATIONS, BACKBONES, DECODERS, NOISE_TYPES, OPTIMIZERS, SCHEDULERS, DenoisingAutoEncoder,
                   EncoderClassifier, SupervisedEncoder)

# --mode, one of:
# pretrain: DenoisingAutoEncoder (--noise_level 0 = the plain AE baseline).
# pretrain_supervise_encoder: SupervisedEncoder, the same encoder trained with labels.
# linear_probe, finetune: EncoderClassifier on a saved encoder.
# knn, analysis: no training, evaluation of a saved encoder's h.
MODES = (
    "pretrain",
    "pretrain_supervise_encoder",
    "linear_probe",
    "finetune",
    "knn",
    "analysis",
)
DATASETS = {"tinyimagenet": "TinyImageNet", "cifar100": "CIFAR100"}
# --encoder, the encoder the evaluation modes load:
# dae / supervised: the file pretrain / pretrain_supervise_encoder saved with the same settings.
# random: no file, a random-init encoder (baseline).
ENCODERS = ("dae", "supervised", "random")


def architecture_tag(encoder):
    """The architecture part of a file name: empty for the defaults (small encoder, light
    decoder), otherwise "_resnet18", and "_resnet18dec" for a DAE with the ResNet-18 decoder."""
    tag = "" if config.BACKBONE == "small" else f"_{config.BACKBONE}"
    if encoder == "dae" and config.DECODER != "light":
        tag += f"_{config.DECODER}dec"
    return tag


def encoder_path(encoder, pretrain_dataset, writing=False):
    """File name of a saved encoder: its dataset, hyperparameters, architecture and seed,
    so no run overwrites another's encoder. A search trial gets "_search".
    writing: the run trains this encoder (pretrain); a DAE tuning trial also lists its training settings."""
    if encoder == "random":
        return None
    if encoder == "supervised":
        name = f"supervised_{pretrain_dataset}_{config.LATENT_DIM}_{config.LABEL_FRACTION}{architecture_tag(encoder)}_s{config.TRAIN_SEED}"
        if config.ACTIVATION:
            name += f"_{config.ACTIVATION}"  # loading it needs the same --activation
        name += run_info.dae_tags(("CHANNELS",))  # the same channels as the DAE it is compared with
    else:
        name = f"dae_{pretrain_dataset}_{config.NOISE_TYPE}_{config.NOISE_LEVEL}_{config.LATENT_DIM}{architecture_tag(encoder)}_s{config.TRAIN_SEED}"
        if config.NOISE_TYPE2 != "none":
            name += f"_{config.NOISE_TYPE2}_{config.NOISE_LEVEL2}"  # the second noise
        trial = writing and (config.SEARCH or config.TUNE)
        name += run_info.dae_tags(run_info.DAE_ARCHITECTURE + (run_info.DAE_TRAINING if trial else ()))
    if config.SEARCH:
        name += "_search"
    elif config.TUNE and (encoder == "supervised" or writing):
        name += "_tune"  # a tuning run of an encoder must not overwrite the final one
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
    # The DAE hyperparameters of the search
    parser.add_argument("--noise_type", choices=NOISE_TYPES, default=config.NOISE_TYPE)
    parser.add_argument("--noise_level", type=float, default=config.NOISE_LEVEL)
    parser.add_argument("--patch_size", type=int, default=config.PATCH_SIZE, help="patch_mask: the side of a patch.")
    parser.add_argument("--noise_type2", choices=("none", *NOISE_TYPES), default=config.NOISE_TYPE2,
                        help="A second noise on top of --noise_type.")
    parser.add_argument("--noise_level2", type=float, default=config.NOISE_LEVEL2)
    parser.add_argument("--noise_level_end", type=float, default=config.NOISE_LEVEL_END,
                        help="Rising noise: from --noise_level in the first epoch to this in the last.")
    parser.add_argument("--loss", choices=("mse", "l1"), default=config.DAE_LOSS, help="The DAE's reconstruction loss.")
    parser.add_argument("--latent_dim", type=int, default=config.LATENT_DIM)
    parser.add_argument("--channels", type=int, nargs=2, default=config.CHANNELS, metavar=("C1", "C2"),
                        help="The output channels of the small encoder's two conv layers.")
    parser.add_argument("--scheduler", choices=SCHEDULERS, default=config.DAE_SCHEDULER, help="The DAE's lr schedule.")
    parser.add_argument("--data_fraction", type=float, default=config.DATA_FRACTION,
                        help="pretrain: the share of the training images the DAE trains on (the search's proxy).")
    parser.add_argument("--label_fraction", type=float, default=config.LABEL_FRACTION, help="1.0, 0.1 or 0.01")
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=config.NUM_EPOCHS,
        help="Epochs of the DAE (pretrain), fixed, no early stopping. Default: config.NUM_EPOCHS. "
        "The classifiers use config.TARGET_NUM_EPOCHS as a cap instead.",
    )
    parser.add_argument(
        "--backbone",
        choices=BACKBONES,
        default=config.BACKBONE,
        help="The encoder of every model: small (the course encoder) or resnet18. "
        "linear_probe, finetune, knn and analysis need the backbone their encoder was trained with.",
    )
    parser.add_argument(
        "--decoder",
        choices=DECODERS,
        default=config.DECODER,
        help="The DAE decoder: light or resnet18 (inverted ResNet-18). The modes that load "
        "a DAE encoder need it too, because it is part of that encoder's file name.",
    )
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
        "--finetune_from",
        choices=("layer1", "layer2", "layer3", "layer4", "fc"),
        default=config.FINETUNE_FROM,
        help="finetune with --backbone resnet18 only: partial fine-tuning. The encoder layers before "
        "this one stay frozen; it and the layers after it train with the head (e.g. layer3: layer3, layer4, fc).",
    )
    # The training hyperparameters: of the classifiers (linear_probe, finetune, pretrain_supervise_encoder),
    # or of the DAE with --mode pretrain. None: config.py's value for that model.
    parser.add_argument("--lr", type=float, default=None, help="Learning rate. Default: config.")
    parser.add_argument("--head_lr_mult", type=float, default=config.HEAD_LR_MULT, help="The head's lr / the encoder's.")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--wd", type=float, default=None, help="Weight decay.")
    parser.add_argument("--dropout", type=float, default=None, help="Share of h zeroed before the head (the DAE: the decoder).")
    parser.add_argument("--augment", choices=tuple(AUGMENTS), default=None)
    parser.add_argument("--label_smoothing", type=float, default=config.LABEL_SMOOTHING)
    parser.add_argument("--optimizer", choices=tuple(OPTIMIZERS), default=None)
    parser.add_argument("--activation", choices=tuple(ACTIVATIONS), default=config.ACTIVATION,
                        help="The encoder's activation. The modes that load an encoder need the one it trained with.")
    parser.add_argument("--max_rounds", type=int, default=config.TARGET_NUM_EPOCHS, help="Cap in rounds (config).")
    parser.add_argument("--patience", type=int, default=config.PATIENCE, help="Rounds with no better val accuracy before stopping.")
    parser.add_argument("--tune", action="store_true", help="A tuning run: chosen by val, so no test and no test curve.")
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
    if args.finetune_from is not None and (args.mode != "finetune" or args.backbone != "resnet18"):
        parser.error("--finetune_from needs --mode finetune and --backbone resnet18")
    if args.activation == ("relu" if args.backbone == "resnet18" else "gelu"):
        args.activation = None  # the backbone's own
    config.OUTPUT_DIR = args.output_dir
    config.DATASET = DATASETS[args.dataset]
    config.NOISE_TYPE = args.noise_type
    config.NOISE_LEVEL = args.noise_level
    config.LATENT_DIM = args.latent_dim
    config.PATCH_SIZE = args.patch_size
    config.NOISE_TYPE2 = args.noise_type2
    config.NOISE_LEVEL2 = args.noise_level2
    config.NOISE_LEVEL_END = args.noise_level_end
    config.DAE_LOSS = args.loss
    config.CHANNELS = tuple(args.channels)
    config.DAE_SCHEDULER = args.scheduler
    config.DATA_FRACTION = args.data_fraction
    config.LABEL_FRACTION = args.label_fraction
    config.NUM_EPOCHS = args.num_epochs
    config.BACKBONE = args.backbone
    config.DECODER = args.decoder
    config.TRAIN_SEED = args.seed
    config.SEARCH = args.search
    config.FINETUNE_FROM = args.finetune_from
    # The training flags set the DAE's settings in pretrain, the classifier's otherwise
    names = (("LEARNING_RATE", "DAE_BATCH_SIZE", "WEIGHT_DECAY", "DAE_DROPOUT", "DAE_AUGMENT", "DAE_OPTIMIZER")
             if args.mode == "pretrain" else
             ("CLASSIFIER_LR", "BATCH_SIZE", "CLASSIFIER_WD", "DROPOUT", "AUGMENT", "OPTIMIZER"))
    for name, value in zip(names, (args.lr, args.batch_size, args.wd, args.dropout, args.augment, args.optimizer)):
        if value is not None:
            setattr(config, name, value)
    config.HEAD_LR_MULT = args.head_lr_mult
    config.LABEL_SMOOTHING = args.label_smoothing
    config.ACTIVATION = args.activation
    config.TARGET_NUM_EPOCHS = args.max_rounds
    config.PATIENCE = args.patience
    config.TUNE = args.tune
    if args.tune:
        config.RUN_TAG = "tune"
        config.TEST_CURVE = False
    if args.search:
        config.RUN_TAG = "search"
    # pretrain writes the encoder; the other modes read one (trained on --pretrain_dataset).
    if args.mode == "pretrain":
        config.ENCODER_PATH = encoder_path("dae", config.DATASET, writing=True)
    elif args.mode == "pretrain_supervise_encoder":
        config.ENCODER_PATH = encoder_path("supervised", config.DATASET, writing=True)
    else:
        config.ENCODER_PATH = encoder_path(args.encoder, DATASETS[args.pretrain_dataset])
    return args


def build_model(mode):
    num_classes = NUM_CLASSES[config.DATASET]
    label_smoothing = config.LABEL_SMOOTHING

    if mode == "pretrain":
        return DenoisingAutoEncoder(
            learning_rate=config.LEARNING_RATE,
            latent_dim=config.LATENT_DIM,
            noise_type=config.NOISE_TYPE,
            noise_level=config.NOISE_LEVEL,
            weight_decay=config.WEIGHT_DECAY,
            backbone=config.BACKBONE,
            decoder=config.DECODER,
            activation=config.ACTIVATION,
            channels=config.CHANNELS,
            dropout=config.DAE_DROPOUT,
            optimizer=config.DAE_OPTIMIZER,
            scheduler=config.DAE_SCHEDULER,
            patch_size=config.PATCH_SIZE,
            noise_type2=config.NOISE_TYPE2,
            noise_level2=config.NOISE_LEVEL2,
            noise_level_end=config.NOISE_LEVEL_END,
            loss=config.DAE_LOSS,
        )

    if mode == "pretrain_supervise_encoder":
        return SupervisedEncoder(
            learning_rate=config.CLASSIFIER_LR or config.LEARNING_RATE,
            dropout=config.DROPOUT,
            weight_decay=config.CLASSIFIER_WD,
            optimizer=config.OPTIMIZER,
            activation=config.ACTIVATION,
            num_classes=num_classes,
            latent_dim=config.LATENT_DIM,
            backbone=config.BACKBONE,
            label_smoothing=label_smoothing,
            channels=config.CHANNELS,
        )

    # linear_probe, finetune, knn and analysis: the saved (or random) encoder + a new head.
    return EncoderClassifier(
        learning_rate=config.CLASSIFIER_LR or (config.FINETUNE_LEARNING_RATE if mode == "finetune" else config.LEARNING_RATE),
        dropout=config.DROPOUT,
        weight_decay=config.CLASSIFIER_WD,
        head_lr_mult=config.HEAD_LR_MULT,
        optimizer=config.OPTIMIZER,
        activation=config.ACTIVATION,
        num_classes=num_classes,
        latent_dim=config.LATENT_DIM,
        encoder_path=config.ENCODER_PATH,
        freeze_encoder=(mode != "finetune"),
        backbone=config.BACKBONE,
        label_smoothing=label_smoothing,
        finetune_from=config.FINETUNE_FROM if mode == "finetune" else None,
        channels=config.CHANNELS,
    )
