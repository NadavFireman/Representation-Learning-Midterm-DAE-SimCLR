# Representation Learning Midterm: DAE and SimCLR on Tiny-ImageNet

Midterm assignment of the Representation Learning course (HIT, 2026).
Part A trains a denoising autoencoder (DAE) on Tiny-ImageNet without labels and
evaluates its encoder. Part B (SimCLR) is not here yet.

Built on the course pipeline: the models are LightningModules (`model.py`), the
data is a LightningDataModule (`datamodule.py`), and every run is one call of
`train.py`. The settings are in `config.py`; command-line flags override them.

## Run it

### On Colab (as used for the report)

Two notebooks, in this order, each in Colab with a GPU runtime (L4 or A100):

- `colab_tuning.ipynb`: the hyperparameters of CNN-3 fine-tuning and the ResNet-18 linear
  probe, on val only, on 10% of the labels; the other classifiers keep `config.py`. Three
  grids, one after the other: learning (learning rate, the head's learning rate, batch size),
  regularization (dropout, augmentation, label smoothing), then the optimizer (AdamW, Adam,
  NAdam, RAdam, SGD) and its weight decay. In each, one hyperparameter at a time, 5 values,
  the others fixed at what was chosen so far; the best by val accuracy is kept. Every run
  stops after 3 rounds with no better val accuracy (`--patience 3`). The choices go to
  `midterm_runs/tuning_choice.json`.
- `colab_final.ipynb`: every final run, with those choices. Its cells:

1. Clone the repository (or pull it, if the runtime already has it).
2. Install the requirements and check the GPU.
3. Mount Google Drive. Every output (logs, checkpoints, encoders) goes to
   `midterm_runs` there, so a disconnect loses nothing. After a disconnect, run
   the cells again: an unfinished run continues from its own `last.ckpt`, and a
   finished one (it has `results.json` in its folder) prints its results and is skipped.
4. Start TensorBoard: all the runs in one view.
5. Run the groups: (1) CNN-3, the course's small encoder, (2) ResNet-18 with full
   fine-tuning, (3) ResNet-18 with partial fine-tuning (`--finetune_from layer3`), (4) seeds
   1 and 2 of group 1. Groups 1+4 and 2+3 can run at the same time in two runtimes.
6. The results of every run, as one table.

`history/` keeps the earlier notebooks, among them `colab_checks.ipynb`: the checks made
before the final runs (the AE and the DAE for 30 epochs, the new metrics on the GPU).

### Locally

```bash
pip install torch torchvision          # see pytorch.org for the build that fits your GPU
pip install -r requirements.txt
python train.py --mode pretrain        # trains the DAE with the settings in config.py
```

The datasets download by themselves into `dataset/` on the first run:
Tiny-ImageNet from Hugging Face (`zh-plus/tiny-imagenet`) and CIFAR-100 from torchvision.

## The modes of train.py

| `--mode` | What it does |
|---|---|
| `pretrain` | Trains the DAE on Tiny-ImageNet without labels and saves its encoder. `--noise_level 0` gives the plain autoencoder (the AE baseline). |
| `pretrain_supervise_encoder` | The same encoder trained from scratch with labels (the supervised baseline), on `--label_fraction` of the labels. |
| `linear_probe` | The saved encoder stays frozen; only a new linear head is trained. |
| `finetune` | The saved encoder and a new head are trained together, with a smaller learning rate. |
| `knn` | No training. k-NN accuracy (k = 20, cosine on L2-normalized h, similarity-weighted vote). |
| `analysis` | No training. Per-class k-NN accuracy, t-SNE, nearest-neighbor retrieval and the robustness test (`analysis.py`). |

Other flags:

| Flag | Values |
|---|---|
| `--encoder` | `dae`, `supervised` or `random`: the encoder that `linear_probe`, `finetune`, `knn` and `analysis` load. `random` is the random-init baseline. |
| `--dataset` | `tinyimagenet`, or `cifar100` for transfer. |
| `--label_fraction` | `1.0`, `0.1` or `0.01`: a class-balanced share of the labels, the same images for every model. |
| `--noise_type`, `--noise_level`, `--latent_dim` | The DAE hyperparameters. `--noise_type`: `gaussian`, `mask`, `salt_pepper`, `patch_mask` (a share of the patches, as in MAE) or `block` (one square, as in the context encoder). |
| `--patch_size` | `patch_mask`: the side of a patch (default 8). |
| `--noise_type2`, `--noise_level2` | A second noise on top of `--noise_type` (default `none`). |
| `--noise_level_end` | Rising noise: from `--noise_level` in the first epoch to this in the last, in a straight line (default: none, a fixed level). |
| `--loss` | The DAE's reconstruction loss: `mse` (default) or `l1`. |
| `--channels C1 C2` | The output channels of CNN-3's two conv layers (default 32 64). The light decoder mirrors them, and the modes that load an encoder need the same. |
| `--scheduler` | The DAE's lr schedule: `none` (default) or `cosine`. |
| `--data_fraction` | `pretrain`: the share of the training images the DAE trains on, class-balanced (default 1.0; the DAE search uses 0.1). |
| `--backbone` | `small` (default, CNN-3, the course encoder) or `resnet18`. The modes that load an encoder need the backbone it was trained with. |
| `--decoder` | The DAE decoder: `light` (default) or `resnet18` (inverted ResNet-18). The modes that load a DAE encoder need it too: it is part of the encoder's file name. |
| `--finetune_from` | `finetune` with `--backbone resnet18` only: partial fine-tuning. The encoder layers before this one (`layer1` to `layer4`, or `fc`) stay frozen, with their BatchNorm statistics; it and the layers after it train with the head. The run folder gets `finetune-from-<layer>`. |
| `--lr`, `--head_lr_mult`, `--batch_size`, `--wd` | The optimization: learning rate (default `FINETUNE_LEARNING_RATE` for `finetune`, `LEARNING_RATE` otherwise), the head's learning rate as a multiple of it, batch size, weight decay. With `--mode pretrain` `--lr`, `--batch_size` and `--wd` set the DAE's (defaults `LEARNING_RATE`, `DAE_BATCH_SIZE`, `WEIGHT_DECAY`). |
| `--dropout`, `--augment`, `--label_smoothing` | The regularization: share of h zeroed in training (`nn.Dropout`; before the head, or before the decoder with `--mode pretrain`), the training augmentation (`none`, `flip`, `crop`, `crop_rotate`, `crop_jitter`, `flip_rotate`, `flip_jitter`, `flip_rotate_jitter`; `augmentations.AUGMENTS`; the DAE's default is `flip`), label smoothing. |
| `--optimizer` | The optimizer: `adamw` (default), `adam`, `nadam`, `radam` or `sgd` (Nesterov momentum 0.9). With `--mode pretrain`, the DAE's. |
| `--activation` | The encoder's activation: `relu`, `leaky_relu`, `gelu`, `silu`, `mish` (default: the backbone's own, GELU for CNN-3, ReLU for ResNet-18). An encoder trained with another is saved with its name in the file, and is loaded with the same flag. |
| `--max_rounds` | Cap of a classifier, in rounds (default 100). A round is as many training images as 100% of the labels, so 1% validates every 100 epochs and 10% every 10: every label fraction gets the same budget and the same patience. |
| `--patience` | Rounds with no better val accuracy before a classifier stops (default 5; the tuning uses 3). |
| `--tune` | A tuning run: chosen by val accuracy, so no test and no test curve; the folder has `tune` instead of `final`. |
| `--num_epochs` | Epochs of the DAE (`pretrain`), default `config.NUM_EPOCHS` (12). The classifiers have their own cap, `TARGET_NUM_EPOCHS`. |
| `--seed` | The training seed. The splits and the label subsets always use `config.SEED`. |
| `--search` | One trial of the hyperparameter search: proxy setting (10 epochs, the k-NN every 2 epochs with a bank of 10,000 images), no test set; the folder has `search` instead of `final`. |
| `--output_dir` | Where the logs, checkpoints and encoders go. |

Examples:

```bash
python train.py --mode pretrain --noise_level 0                                   # the AE baseline
python train.py --mode finetune --encoder dae --label_fraction 0.1                # fine-tune with 10% of the labels
python train.py --mode linear_probe --encoder random --dataset cifar100           # random encoder, transfer
python train.py --mode analysis --encoder dae                                     # the A5 evidence of the DAE
python train.py --mode pretrain --backbone resnet18 --decoder resnet18            # ResNet-18 encoder-decoder DAE
python train.py --mode linear_probe --encoder dae --backbone resnet18 --decoder resnet18   # its linear probe
```

## Data splits

- **train**: Tiny-ImageNet train minus the held-out images (90,000).
- **val**: 50 training images per class (10,000), for all tuning and model selection.
- **test**: the official Tiny-ImageNet validation set (10,000), only for the final numbers.

## Final DAE settings

Chosen by validation k-NN accuracy, not by reconstruction error, in two searches:

| Setting | Value |
|---|---|
| Corruption | salt and pepper, 50% of the pixels |
| h (`latent_dim`) | 256, the output of the encoder's last Linear layer |
| Activation | GELU |
| Optimizer | AdamW, lr 1e-3, weight decay 0.01 |
| Batch size | 512 |
| Epochs | 12; the kept checkpoint is the epoch with the best val k-NN |

## Hyperparameter search

- `colab_dae_grids.ipynb`: search 3, five grids one after the other (the noise; the conv channels and h;
  dropout, augmentation and batch size; the optimizer and the activation; the learning rate, weight decay
  and scheduler), every combination in each, the winner carried to the next grid, by val k-NN. Each trial
  trains on 10% of the training images for 10 epochs; the 5 best then train on all of them for 20 epochs.
  `colab_dae_grids_2.ipynb` continues it with grid 6 (a second noise on top of the winner, combined
  augmentations) and grid 7 (smaller conv 2 and patch size).
- `history/colab_experiments.ipynb`: search 1, staged grids over the corruption, its level,
  `latent_dim`, learning rate, batch size and more (run on an A100).
- `history/colab_search2.ipynb`: search 2, five grids of 2 to 4 hyperparameters each, repeated
  until no grid changes the best settings, then the top 3 with two more seeds (run on an L4).

## Files

| File | Contents |
|---|---|
| `train.py` | One run: builds the model and the data, trains, validates and tests. |
| `run_info.py` | The run folder name (date, time, model, what) and `run_info.txt`. |
| `config.py` | All the settings. |
| `utils.py` | Command-line flags, encoder file names, building the model for each mode. |
| `model.py` | The encoders (small, ResNet-18), the decoders (light, inverted ResNet-18), the DAE, the classifier (linear probe and fine-tuning) and the supervised encoder. |
| `datamodule.py` | Tiny-ImageNet and CIFAR-100, the splits and the label subsets. |
| `augmentations.py` | The transforms and the Tiny-ImageNet normalization. |
| `knn.py` | k-NN evaluation. |
| `callbacks.py` | Per-epoch k-NN, per-epoch metrics, progress bar. |
| `analysis.py` | Per-class accuracy, t-SNE, retrieval and robustness. |
| `colab_dae_grids.ipynb` | Launcher: the DAE hyperparameters (search 3), five grids on val k-NN. |
| `colab_dae_grids_2.ipynb` | Launcher: search 3 continued, grid 6 (a second noise, combined augmentations) and grid 7 (smaller conv 2 and patch). |
| `colab_tuning.ipynb` | Launcher: the classifier hyperparameters, three grids on val. |
| `colab_final.ipynb` | Launcher: every final run, with the tuning's choices. |

## Metrics and epochs

- Classifiers (`linear_probe`, `finetune`, `pretrain_supervise_encoder`) log top-1 and
  top-5 accuracy, and on val and test also macro precision, recall, F1 and AUROC
  (torchmetrics). After the test they save the confusion matrix (`confusion_matrix.csv`,
  `.png`) and the 20 most confused class pairs (`most_confused.csv`). `analysis` saves
  the same for the k-NN (`knn_*`).
- Every run that trains (not `--search`) also scores the test set after every epoch,
  with the same metrics, and k-NN for the encoders (`config.TEST_CURVE`). TensorBoard
  shows them as `epoch/test_*`, next to `epoch/train_*` and `epoch/val_*`. Only val
  chooses the checkpoint and stops a run, so the test curve is for plots only and
  the test numbers stay final numbers.
- Every classifier stops on `val_accuracy` after 5 rounds with no better one, with a cap
  of 100 rounds, and logs its calibration error (ECE) too. The DAE trains a fixed number of epochs (`--num_epochs`, default 12) and
  keeps the epoch with the best val k-NN.
- `finetune --encoder supervised --dataset cifar100 --label_fraction f` loads the
  supervised encoder trained on the same fraction f of the Tiny-ImageNet labels.

## Logs

Every run writes to its own folder in `<output_dir>`:
`date<DDMMYYYY>_time<HHMMSS>_<model>_final_<stage>_<what>_seed<seed>`, for example
`date23112026_time101500_ResNet18_final_finetune_DAE_on_TinyImageNet_10pct_seed42`. The date and
time are those of its first start; after a disconnect the run finds the same folder and
continues. The folder holds the TensorBoard logs, every setting (`hparams.yaml`), the
results (`results.json`) and `run_info.txt`: what the run is, its hyperparameters, epochs,
results, start and end time and training time (also in TensorBoard's TEXT tab). The
encoders go to `<output_dir>/encoders`. To view the logs:

```bash
tensorboard --logdir <output_dir>
```
