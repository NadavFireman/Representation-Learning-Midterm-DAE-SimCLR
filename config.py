import os

# Training hyperparameters. The DAE values are the winner of the search, chosen by val k-NN.
LEARNING_RATE = 0.001
BATCH_SIZE = 256  # linear_probe, finetune and the supervised encoder
DAE_BATCH_SIZE = 512  # pretrain (the DAE)
WEIGHT_DECAY = 0.01  # AdamW weight decay of the DAE
NUM_EPOCHS = 12  # the DAE only, no early stopping: its checkpoint is chosen by val k-NN
SWEEP_NUM_EPOCHS = 10  # proxy setting of a search trial (--search)
DATA_FRACTION = 1.0  # --data_fraction: the share of the training images the DAE trains on (class-balanced)
LATENT_DIM = 256  # length of h
NOISE_TYPE = "salt_pepper"  # DAE corruption: model.NOISE_TYPES
NOISE_LEVEL = 0.5  # gaussian: noise std; the others: fraction of pixels, patches or image. 0 = plain AE
PATCH_SIZE = 8  # patch_mask: the side of a patch
DAE_DROPOUT = 0.0  # share of h zeroed before the decoder, in training
DAE_AUGMENT = "flip"  # the DAE's training images (augmentations.AUGMENTS)
DAE_OPTIMIZER = "adamw"  # model.OPTIMIZERS
DAE_SCHEDULER = "none"  # "none" or "cosine"
# Architecture (see model.py)
BACKBONE = "small"  # the encoder: "small" (the course encoder) or "resnet18"
CHANNELS = (32, 64)  # the output channels of the small encoder's two conv layers (--channels)
DECODER = "light"  # the DAE decoder: "light" or "resnet18" (inverted ResNet-18)

# The grid of the A2 search (train.py --search, one trial per call).
# Each trial adds one row to sweep_results.csv; the best is chosen by val k-NN.
SWEEP = {
    "noise_type": ["gaussian", "mask", "salt_pepper"],
    "noise_level": [0.25, 0.5, 0.75],
    "latent_dim": [64, 128, 256],
}

# k-NN evaluation (the assignment): k=20, cosine on L2-normalized h, similarity-weighted vote
KNN_K = 20
KNN_BANK_SIZE = None  # memory bank size. None = all 90,000 training images
SEARCH_KNN_BANK_SIZE = 10000  # --search only: 50 images per class, faster
SEARCH_KNN_EVERY = 2  # --search only: the k-NN every 2 epochs

# Dataset
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")  # next to this file
# Logs, checkpoints and encoders. On Colab, --output_dir points it at Google Drive.
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))
NUM_WORKERS = min(8, os.cpu_count() or 2)  # free Colab has 2 cores, an L4 runtime has more
EVAL_BATCH_SIZE = 1024  # val, test and the k-NN bank: no gradients, so a larger batch is faster
SEED = 42  # data seed: the val split and the label subsets, the same for every model
TRAIN_SEED = 42  # training seed: initialization, data order, noise (--seed)
SEARCH = False  # set by --search
RUN_TAG = "final"  # in every run folder name, after the model: these runs against the earlier ones
# Runs that train also score the test set after every epoch (epoch/test_*), to plot only.
# Only val chooses the checkpoint and stops the run.
TEST_CURVE = True

# Compute related
ACCELERATOR = "auto"  # GPU on Colab
DEVICES = 1
PRECISION = "16-mixed"  # the assignment asks for mixed precision
LIMIT_BATCHES = 1.0  # fraction of the batches per epoch; an int means that many batches

# The modes of train.py (--mode), see utils.MODES
SAVE_ENCODER_MODES = ("pretrain", "pretrain_supervise_encoder")  # these write an encoder file
MODEL_NAME = "pretrain"

DATASET = "TinyImageNet"  # set by --dataset; CIFAR100 is the transfer set
LABEL_FRACTION = 1.0  # class-balanced part of train: 1.0, 0.1 or 0.01
# The classifiers count in rounds: one round is as many training images as 100% of the labels
# (90,000), so 1% validates every 100 epochs and 10% every 10: every label fraction gets the
# same training budget and the same patience. TARGET_NUM_EPOCHS is the cap in rounds (--max_rounds).
TARGET_NUM_EPOCHS = 100
# The classifiers stop on and keep their best val_accuracy
PATIENCE = 5  # rounds without a better val_accuracy before stopping
LABEL_SMOOTHING = 0.1  # 0.9 on the true class, 0.1 spread over all classes
FINETUNE_LEARNING_RATE = 0.0001  # smaller lr, so we don't destroy the pretrained features
# The classifier hyperparameters that the tuning notebook searches (colab_tuning.ipynb)
CLASSIFIER_LR = None  # --lr: None = FINETUNE_LEARNING_RATE for finetune, LEARNING_RATE for the others
HEAD_LR_MULT = 1.0  # --head_lr_mult: the head's lr is this times the encoder's
CLASSIFIER_WD = 0.0  # --wd: AdamW weight decay of the classifiers (0 = Adam)
DROPOUT = 0.0  # --dropout: share of h zeroed before the head, in training
AUGMENT = "flip"  # --augment: the training images of a classifier (augmentations.AUGMENTS)
OPTIMIZER = "adamw"  # --optimizer of the classifiers (model.OPTIMIZERS)
ACTIVATION = None  # --activation of the encoder (model.ACTIVATIONS); None: the backbone's own
TUNE = False  # --tune: a tuning run: val only, no test
# Partial fine-tuning (ResNet-18, --finetune_from): the first layer that trains.
# "layer3" trains layer3, layer4, fc and the head. None = the whole encoder trains.
FINETUNE_FROM = None
ENCODER_PATH = None  # set by utils.parse_args (None = random-init encoder)
