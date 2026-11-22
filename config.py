import os

# Full version: Colab, GPU, all the data.

# Training hyperparameters (defaults for one run; the search below overrides the DAE ones)
# The DAE values are the winner of search 2 (colab_search2.ipynb), chosen by val k-NN.
LEARNING_RATE = 0.001
BATCH_SIZE = 256  # linear_probe, finetune and the supervised encoder
DAE_BATCH_SIZE = 512  # pretrain (the DAE)
WEIGHT_DECAY = 0.01  # AdamW weight decay of the DAE
NUM_EPOCHS = 12  # the DAE (pretrain) only. Fixed, no early stopping: its checkpoint is chosen by val k-NN. Search 2: k-NN peaks at epoch 3-5, then falls
SWEEP_NUM_EPOCHS = 8  # proxy setting for the search (--search): every trial, same number of epochs. Full NUM_EPOCHS only for final models
LATENT_DIM = 256  # length of h. The DAE and EncoderClassifier must use the same value.
NOISE_TYPE = "salt_pepper"  # DAE corruption: "gaussian", "mask" or "salt_pepper" (see model.py)
NOISE_LEVEL = 0.5  # gaussian: noise std; mask / salt_pepper: fraction of pixels. 0 = plain autoencoder (AE baseline)
# Architecture (see model.py). The defaults are the course's small encoder and its mirror decoder.
BACKBONE = "small"  # the encoder of every model: "small" or "resnet18"
DECODER = "light"  # the DAE decoder: "light" (mirror of the small encoder) or "resnet18" (inverted ResNet-18)

# The A2 search through train.py --search: a full grid over these three DAE
# hyperparameters, the other settings as above. train.py does one run per
# call, as in the course; the launcher (the search cell of the notebook) runs
# it once per combination, for example:
#   python train.py --mode pretrain --search --noise_type mask --noise_level 0.25 --latent_dim 64
# Each trial adds one row to sweep_results.csv: val reconstruction loss and val
# k-NN accuracy. The best trial is chosen by k-NN, not by reconstruction.
# The grid holds the final settings (salt_pepper, 0.5, 256), so the chosen model
# is one of the points of the A2 plot. (The first two searches, over more
# settings, ran inside colab_experiments.ipynb and colab_search2.ipynb.)
SWEEP = {
    "noise_type": ["gaussian", "mask", "salt_pepper"],
    "noise_level": [0.25, 0.5, 0.75],
    "latent_dim": [64, 128, 256],
}

# k-NN evaluation, fixed by the assignment: k=20, cosine on L2-normalized h,
# similarity-weighted vote.
KNN_K = 20
KNN_BANK_SIZE = None  # memory bank size for the per-epoch val k-NN. None = all 90,000 training images
# --search only: a class-balanced bank of 100 images per class, as in the first two
# searches. Faster, and the trials stay comparable to them. Final runs use all of train.
SEARCH_KNN_BANK_SIZE = 20000

# Dataset
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")  # next to this file, whatever folder train.py is run from
# Where every run writes its logs (tb_logs), checkpoints, encoders and sweep_results.csv.
# Default: next to this file. On Colab, --output_dir points it at Google Drive, so a
# disconnect loses nothing (the code and the dataset stay on the Colab machine).
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
# Free Colab has 2 CPU cores; an L4 runtime has more, and with 2 workers loading the
# images was the bottleneck. The runs before this change used 2 workers: the
# random flips come in another order now, so a rerun does not repeat them to the digit.
NUM_WORKERS = min(8, os.cpu_count() or 2)
SEED = 42  # data seed: the val split and the label subsets. Fixed, so every model sees the same images
TRAIN_SEED = 42  # training seed: initialization, data order, noise. --seed changes it (repeat runs)
SEARCH = False  # set by --search: one trial of the hyperparameter search (see utils.parse_args)
RETEST = False  # set by --retest: test a finished classifier run again from its best checkpoint, no training (see train.py)

# Compute related
ACCELERATOR = "auto"  # GPU on Colab
DEVICES = 1
PRECISION = "16-mixed"  # the assignment asks for mixed precision
LIMIT_BATCHES = 1.0  # fraction of the batches used per epoch (1.0 = all of them); an int means that many batches

# One model name. Choose from the command line:
#   python train.py --mode pretrain
#   python train.py --mode linear_probe --label_fraction 0.1
#   python train.py --mode finetune --dataset cifar100
#   python train.py --mode pretrain --noise_type mask --noise_level 0.5
# --mode overrides MODEL_NAME with one of these (see model.py):
#   "pretrain"                    : DenoisingAutoEncoder - reconstruction of the clean image, saves the encoder
#   "pretrain_supervise_encoder"  : SupervisedEncoder - the same encoder + head from scratch, with labels
#   "linear_probe"                : EncoderClassifier - that encoder FROZEN + a new head
#   "finetune"                    : EncoderClassifier - that encoder + a new head, all weights
#   "knn"                         : no training - k-NN accuracy of that encoder's h
PRETRAIN_MODES = ("pretrain", "pretrain_supervise_encoder")
# These write the encoder file that linear_probe, finetune and knn load. A later run overwrites it.
SAVE_ENCODER_MODES = ("pretrain", "pretrain_supervise_encoder")
MODEL_NAME = "pretrain"

# Set by --dataset in utils.parse_args. Default is Tiny-ImageNet; cifar100 is the transfer set.
DATASET = "TinyImageNet"
LABEL_FRACTION = 1.0  # class-balanced part of train for linear_probe, finetune and supervised: 1.0, 0.1 or 0.01
# A cap for every classifier run (linear_probe, finetune and the supervised encoder):
# EarlyStopping on val_accuracy (PATIENCE) ends them sooner. With a cap of 30 (12 for
# the supervised encoder), 18 runs stopped while val_accuracy was still rising.
TARGET_NUM_EPOCHS = 100
# How the runs that train a classifier (pretrain_supervise_encoder, linear_probe,
# finetune) stop and pick their checkpoint. --protocol overrides it.
#   "acc":  by val_accuracy (max), patience 5, cross-entropy with label smoothing.
#           Their run folders (and the supervised encoder files) end in "_acc".
#   "loss": by val_loss (min), patience 3, plain cross-entropy: the first runs.
# Why "acc": with "loss" the supervised encoder kept epoch 0-1. val_loss rises as
# soon as the model grows overconfident, while val_accuracy still rises for 2-3
# more epochs (in 29 of 32 runs). Label smoothing keeps that overconfidence down.
PROTOCOL = "acc"
PATIENCE = 5  # "acc": epochs without a better val_accuracy before stopping ("loss": Lightning's default, 3)
LABEL_SMOOTHING = 0.1  # "acc": the target is 0.9 on the true class and 0.1 spread over all classes ("loss": 0)
FINETUNE_LEARNING_RATE = 0.0001  # smaller lr, so we don't destroy the pretrained features
ENCODER_PATH = None  # set by utils.parse_args from the pretrain settings and --encoder (None = random-init encoder)
