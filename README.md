# Representation Learning Midterm: DAE and SimCLR on Tiny-ImageNet

Midterm assignment of the Representation Learning course (HIT, 2026).
Part A trains a denoising autoencoder (DAE) on Tiny-ImageNet without labels and
evaluates its encoder. Part B (SimCLR) is not here yet.

Built on the course pipeline: the models are LightningModules (`model.py`), the
data is a LightningDataModule (`datamodule.py`), and every run is one call of
`train.py`. The settings are in `config.py`; command-line flags override them.

## Run it

### On Colab (as used for the report)

Open `colab_train.ipynb` in Colab with a GPU runtime and run the cells in order:

1. Clone the repository (or pull it, if the runtime already has it).
2. Install the requirements and check the GPU.
3. Mount Google Drive. Every output (logs, checkpoints, encoders) goes there, so a
   disconnect loses nothing. After a disconnect, run the cells again: an unfinished
   run continues from its own `last.ckpt`, and a finished one (it has `results.json`
   in its folder) prints its results and is skipped.
4. Start TensorBoard.
5. Run one `train.py` command, or
6. run all the runs of Part A, one after the other (43 runs).

`colab_resnet.ipynb` (L4 GPU) runs the same protocol with a ResNet-18 encoder
(`--backbone resnet18`): stage 1 compares two DAE decoders, the light one and an
inverted ResNet-18 (`--decoder resnet18`); stage 2 runs the full protocol with the
chosen decoder.

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
| `--noise_type`, `--noise_level`, `--latent_dim` | The DAE hyperparameters. |
| `--backbone` | `small` (default, the course encoder) or `resnet18`. The modes that load an encoder need the backbone it was trained with. |
| `--decoder` | The DAE decoder: `light` (default) or `resnet18` (inverted ResNet-18). The modes that load a DAE encoder need it too: it is part of the encoder's file name. |
| `--protocol` | `acc` (default): classifier runs stop on and keep the best `val_accuracy`, with patience 5 and label smoothing 0.1; their folders and the supervised encoder files end in `_acc`. `loss`: the first runs, by `val_loss`. |
| `--num_epochs` | Epochs of the DAE (`pretrain`), default `config.NUM_EPOCHS` (12). The classifiers have their own cap, `TARGET_NUM_EPOCHS`. |
| `--seed` | The training seed. The splits and the label subsets always use `config.SEED`. |
| `--search` | One trial of the hyperparameter search: proxy setting (8 epochs, a k-NN bank of 20,000 images), no test set. |
| `--retest` | A finished `linear_probe`, `finetune` or `pretrain_supervise_encoder` run: no training, its best checkpoint is tested again and the new metrics are added to its `results.json`; the first `results.json` is kept as `results_before_retest.json`, and if `test_accuracy` comes out different (by more than 0.0005) nothing is written. `analysis` simply runs again (checked on `test_knn_top1`). A run with no `results.json` is an error, never a training run. |
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

- `colab_experiments.ipynb`: search 1, staged grids over the corruption, its level,
  `latent_dim`, learning rate, batch size and more (run on an A100).
- `colab_search2.ipynb`: search 2, five grids of 2 to 4 hyperparameters each, repeated
  until no grid changes the best settings, then the top 3 with two more seeds (run on an L4).

## Files

| File | Contents |
|---|---|
| `train.py` | One run: builds the model and the data, trains, validates and tests. |
| `config.py` | All the settings. |
| `utils.py` | Command-line flags, encoder file names, building the model for each mode. |
| `model.py` | The encoders (small, ResNet-18), the decoders (light, inverted ResNet-18), the DAE, the classifier (linear probe and fine-tuning) and the supervised encoder. |
| `datamodule.py` | Tiny-ImageNet and CIFAR-100, the splits and the label subsets. |
| `augmentations.py` | The transforms and the Tiny-ImageNet normalization. |
| `knn.py` | k-NN evaluation. |
| `callbacks.py` | Per-epoch k-NN, per-epoch metrics, progress bar. |
| `analysis.py` | Per-class accuracy, t-SNE, retrieval and robustness. |

## Metrics and epochs

- Classifiers (`linear_probe`, `finetune`, `pretrain_supervise_encoder`) log top-1 and
  top-5 accuracy, and on val and test also macro precision, recall, F1 and AUROC
  (torchmetrics). After the test they save the confusion matrix (`confusion_matrix.csv`,
  `.png`) and the 20 most confused class pairs (`most_confused.csv`). `analysis` saves
  the same for the k-NN (`knn_*`).
- Every classifier stops on `val_accuracy` (patience 5), with a cap of 100 epochs. The
  DAE trains a fixed 12 epochs and keeps the epoch with the best val k-NN.
- `finetune --encoder supervised --dataset cifar100 --label_fraction f` loads the
  supervised encoder trained on the same fraction f of the Tiny-ImageNet labels.
- The runs made before the cap was raised (30 epochs, 12 for the supervised encoder)
  used 2 data-loading workers; a rerun of them now uses up to 8, so its random flips,
  and the last digits of its numbers, differ.

## Logs

Every run writes TensorBoard logs to `<output_dir>/tb_logs/<run name>/`, with its
settings in `hparams.yaml`. To view them:

```bash
tensorboard --logdir <output_dir>/tb_logs
```
