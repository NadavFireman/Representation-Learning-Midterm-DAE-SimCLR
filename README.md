# Representation Learning Midterm: DAE and SimCLR on Tiny-ImageNet

Midterm assignment of the Representation Learning course (HIT, 2026).
Part A trains a denoising autoencoder (DAE) on Tiny-ImageNet without labels and
evaluates its encoder. Part B (SimCLR) is not here yet.

Built on the course pipeline: the models are LightningModules (`model.py`), the
data is a LightningDataModule (`datamodule.py`), and every run is one call of
`train.py`. The settings are in `config.py`; command-line flags override them.

## Run it

### On Colab (as used for the report)

Open `colab_train_MA_A.ipynb`
([open in Colab](https://colab.research.google.com/github/NadavFireman/Representation-Learning-Midterm-DAE-SimCLR/blob/main/colab_train_MA_A.ipynb))
with a GPU runtime (L4) and run all the cells. The notebook only clones the repository,
installs it, mounts Google Drive, starts TensorBoard and calls `train.py`, one call per run:
47 runs, about 6 to 8 hours.

Every output goes to the folder `midterm_final` on Drive. `DRIVE_DIR` in the Drive cell is
the author's folder: it is created if it does not exist, or change it to your own. After a
disconnect, run the cells again: a finished run (it has `results.json`) is skipped, and an
unfinished one continues from its `last.ckpt`.

### Locally

```bash
pip install torch torchvision          # see pytorch.org for the build that fits your GPU
pip install -r requirements.txt
```

The datasets download by themselves into `dataset/` on the first run (Tiny-ImageNet from
Hugging Face, `zh-plus/tiny-imagenet`, and CIFAR-100 from torchvision). The runs of the
report are these calls; the notebook loops over the encoders, the datasets and the label
fractions:

```bash
python train.py --mode pretrain                                              # the DAE; saves its encoder
python train.py --mode pretrain --noise_level 0                              # the AE baseline
python train.py --mode knn --encoder dae                                     # k-NN on the test set
python train.py --mode linear_probe --encoder dae
python train.py --mode analysis --encoder dae                                # robustness, per-class accuracy, t-SNE, retrieval
python train.py --mode finetune --encoder dae --label_fraction 0.1           # also 0.01 and 1.0
python train.py --mode pretrain_supervise_encoder --label_fraction 0.1       # the supervised baseline; also 0.01 and 1.0
```

## The modes and flags of train.py

| `--mode` | What it does |
|---|---|
| `pretrain` | Trains the DAE on Tiny-ImageNet without labels and saves its encoder. |
| `pretrain_supervise_encoder` | The same encoder trained from scratch with labels (the supervised baseline). |
| `linear_probe` | The saved encoder stays frozen; only a new linear head is trained. |
| `finetune` | The saved encoder and a new head are trained together. |
| `knn` | No training. k-NN accuracy (k = 20, cosine on L2-normalized h, similarity-weighted vote). |
| `analysis` | No training. Per-class k-NN accuracy, t-SNE, nearest-neighbor retrieval and the robustness test. |

| Flag | Values |
|---|---|
| `--encoder` | `dae`, `supervised` or `random`: the encoder that the evaluation modes load. |
| `--noise_level` | The corruption level of the DAE. `0` is the plain autoencoder; add it to load the AE too (`--encoder dae --noise_level 0`). |
| `--dataset` | `tinyimagenet`, or `cifar100` for transfer. |
| `--label_fraction` | `1.0`, `0.1` or `0.01`: a class-balanced share of the labels, the same images for every model. |
| `--seed` | The training seed (default 42). The splits and the label subsets always use `config.SEED`. |
| `--output_dir` | Where the logs, checkpoints and encoders go (default: next to the code). |
| `--search`, `--tune` | A tuning run: scored on val only, never on the test set. |

`python train.py --help` lists every flag (the DAE and classifier hyperparameters, and the
optional ResNet-18 backbone).

## Data splits

- **train**: Tiny-ImageNet train minus the held-out images (90,000).
- **val**: 50 training images per class (10,000), for all tuning and model selection.
- **test**: the official Tiny-ImageNet validation set (10,000), only for the final numbers.

CIFAR-100 (the transfer set) is upscaled to 64×64 and split the same way (45,000 / 5,000 /
10,000). Both datasets are normalized with the mean and standard deviation of Tiny-ImageNet.

## Final settings

| Run | Settings |
|---|---|
| DAE | CNN-3 encoder (two 3×3 conv layers, 32 and 64 channels, stride 2, GELU, and a Linear layer); h is its output, 256 numbers. Salt and pepper on 50% of the pixels, dropout 0.4 on h, AdamW, lr 1e-3, weight decay 0.01, batch 512, horizontal flip, MSE, 20 epochs. The kept checkpoint is the epoch with the best val k-NN. |
| Linear probe | AdamW, lr 1e-3, batch 256, flip, label smoothing 0.1. |
| Fine-tuning and the supervised baseline | The recipe tuned on val (`config.TUNED`): SGD with Nesterov momentum, lr 0.003, weight decay 1e-5, dropout 0.5, `crop_jitter`, batch 256. In fine-tuning the new head's learning rate is 3 times the encoder's. |

Every classifier stops after 5 rounds with no better val accuracy (a round is as many
training images as 100% of the labels), with a cap of 100 rounds.

## Hyperparameter search

Every choice was made on val, by k-NN accuracy for the DAE and not by its reconstruction
error. The searches were small grids run one after the other, the winner of each carried
to the next. Their launchers were Colab notebooks that only looped over calls like these,
and are not part of the repository:

```bash
# the DAE, proxy setting (221 trials): 10% of the training images, 10 epochs
python train.py --mode pretrain --search --data_fraction 0.1 --noise_type patch_mask --noise_level 0.5
# the DAE on all the training images (47 trials), then the three best with two more seeds
python train.py --mode pretrain --tune --num_epochs 20 --noise_type salt_pepper --noise_level 0.5 --dropout 0.4
# the fine-tuning recipe (33 runs): one hyperparameter at a time, on 10% of the labels
python train.py --mode finetune --encoder dae --tune --label_fraction 0.1 --patience 3 --max_rounds 50 --lr 3e-4
```

## Files

| File | Contents |
|---|---|
| `train.py` | One run: builds the model and the data, trains, validates and tests. |
| `config.py` | All the settings. |
| `utils.py` | Command-line flags, encoder file names, building the model for each mode. |
| `model.py` | The encoders, the decoders, the DAE, the classifier and the supervised encoder. |
| `datamodule.py` | Tiny-ImageNet and CIFAR-100, the splits and the label subsets. |
| `augmentations.py` | The transforms and the Tiny-ImageNet normalization. |
| `knn.py` | k-NN evaluation. |
| `callbacks.py` | Per-epoch k-NN, per-epoch metrics, progress bar. |
| `analysis.py` | Per-class accuracy, t-SNE, retrieval and robustness. |
| `run_info.py` | The run folder name and `run_info.txt`. |
| `colab_train_MA_A.ipynb` | Launcher: every final run of Part A. |
| `requirements.txt` | The packages, next to torch and torchvision. |

## Logs

The logs of every run are shared on Google Drive (TensorBoard events, settings and results):

- The 47 final runs: [midterm_final](https://drive.google.com/drive/folders/1r4T1nkcCEf8XEsXHHZT5ghV8tsOcA9ow?usp=sharing)
- The tuning runs: [midterm_runs](https://drive.google.com/drive/folders/1abQOyFbXNlbTOeBvXiSFEFRY8sn7_HuF?usp=sharing)

Every run has its own folder, for example
`date25112026_time123303_CNN3_final_analysis_DAE_on_TinyImageNet_seed42`, with its
TensorBoard logs, every setting (`hparams.yaml`), the results (`results.json`) and
`run_info.txt`. The saved encoders are in `encoders/`. To view the logs:

```bash
tensorboard --logdir <output_dir>
```
