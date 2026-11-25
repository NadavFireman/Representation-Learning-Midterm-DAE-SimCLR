# Representation Learning Midterm: DAE and SimCLR on Tiny-ImageNet

Midterm assignment of the Representation Learning course (HIT, 2026).
Part A trains a denoising autoencoder (DAE) on Tiny-ImageNet without labels and
evaluates its encoder. Part B (SimCLR) is not here yet.

Built on the course pipeline: the models are LightningModules (`model.py`), the
data is a LightningDataModule (`datamodule.py`), and every run is one call of
`train.py`. The settings are in `config.py`; command-line flags override them.

## Run it

### On Colab (as used for the report)

One launcher notebook, `colab_train_MA_A.ipynb`
([open in Colab](https://colab.research.google.com/github/NadavFireman/Representation-Learning-Midterm-DAE-SimCLR/blob/main/colab_train_MA_A.ipynb)),
with a GPU runtime (L4). It only clones the repository, installs it and calls `train.py`,
one call per run. Its cells:

1. Clone the repository (or pull it, if the runtime already has it).
2. Install the requirements and check the GPU.
3. Mount Google Drive. Every output (logs, checkpoints, encoders) goes to the folder
   `midterm_final` there, so a disconnect loses nothing. `DRIVE_DIR` in this cell is the
   author's folder; the folder is created if it does not exist, or change it to your own.
   After a disconnect, run the cells again: an unfinished run continues from its own
   `last.ckpt`, and a finished one (it has `results.json` in its folder) prints its results
   and is skipped.
4. Start TensorBoard: all the runs in one view.
5. The settings of all the runs: the backbone (CNN-3, the small encoder of the course) and
   the seed (42).
6. The runs, in four groups:
   1. **Pretrain**: the DAE and the AE (the same, with `--noise_level 0`).
   2. **Frozen evaluation** of the DAE, the AE and a random encoder: k-NN, linear probe and
      analysis, on Tiny-ImageNet and on CIFAR-100.
   3. **Fine-tuning** of the same three encoders on 1%, 10% and 100% of the labels, on both
      datasets.
   4. **Supervised baseline**: the same encoder trained from scratch with labels on 1%, 10%
      and 100%, and the frozen evaluation of the one trained on 100%.

47 runs, about 6 to 8 hours on an L4.

### Locally

```bash
pip install torch torchvision          # see pytorch.org for the build that fits your GPU
pip install -r requirements.txt
python train.py --mode pretrain        # trains the DAE with the settings in config.py
```

The datasets download by themselves into `dataset/` on the first run:
Tiny-ImageNet from Hugging Face (`zh-plus/tiny-imagenet`) and CIFAR-100 from torchvision.
The logs, checkpoints and encoders go next to the code, or to `--output_dir`.

The runs of the report are these calls (the notebook loops over the encoders, the two
datasets and the label fractions):

```bash
python train.py --mode pretrain                                              # the DAE; saves its encoder
python train.py --mode pretrain --noise_level 0                              # the AE baseline
python train.py --mode knn --encoder dae                                     # k-NN on the test set
python train.py --mode linear_probe --encoder dae
python train.py --mode analysis --encoder dae                                # robustness, per-class accuracy, t-SNE, retrieval
python train.py --mode finetune --encoder dae --label_fraction 0.1           # also 0.01 and 1.0
python train.py --mode pretrain_supervise_encoder --label_fraction 0.1       # the supervised baseline; also 0.01 and 1.0
```

For the AE use `--encoder dae --noise_level 0`, for the random encoder `--encoder random`,
for the supervised one `--encoder supervised`, and for transfer add `--dataset cifar100`.

## The modes of train.py

| `--mode` | What it does |
|---|---|
| `pretrain` | Trains the DAE on Tiny-ImageNet without labels and saves its encoder. `--noise_level 0` gives the plain autoencoder (the AE baseline). |
| `pretrain_supervise_encoder` | The same encoder trained from scratch with labels (the supervised baseline), on `--label_fraction` of the labels. |
| `linear_probe` | The saved encoder stays frozen; only a new linear head is trained. |
| `finetune` | The saved encoder and a new head are trained together, with the recipe tuned on val (`config.TUNED`). |
| `knn` | No training. k-NN accuracy (k = 20, cosine on L2-normalized h, similarity-weighted vote). |
| `analysis` | No training. Per-class k-NN accuracy, t-SNE, nearest-neighbor retrieval and the robustness test (`analysis.py`). |

Other flags:

| Flag | Values |
|---|---|
| `--encoder` | `dae`, `supervised` or `random`: the encoder that `linear_probe`, `finetune`, `knn` and `analysis` load. `random` is the random-init baseline. |
| `--dataset` | `tinyimagenet`, or `cifar100` for transfer. |
| `--pretrain_dataset` | Where the loaded encoder was trained (default `tinyimagenet`). Not used by the pretrain modes. |
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
| `--lr`, `--head_lr_mult`, `--batch_size`, `--wd` | The optimization: learning rate, the head's learning rate as a multiple of it, batch size, weight decay. Without a flag a run uses `config.py`: the DAE's values with `--mode pretrain`, the tuned recipe (`config.TUNED`) for `finetune` and `pretrain_supervise_encoder`, and the classifier defaults for the linear probe. |
| `--dropout`, `--augment`, `--label_smoothing` | The regularization: share of h zeroed in training (`nn.Dropout`; before the head, or before the decoder with `--mode pretrain`), the training augmentation (`none`, `flip`, `crop`, `crop_rotate`, `crop_jitter`, `flip_rotate`, `flip_jitter`, `flip_rotate_jitter`; `augmentations.AUGMENTS`; the DAE's default is `flip`), label smoothing. |
| `--optimizer` | The optimizer: `adamw`, `adam`, `nadam`, `radam` or `sgd` (Nesterov momentum 0.9). With `--mode pretrain`, the DAE's. |
| `--activation` | The encoder's activation: `relu`, `leaky_relu`, `gelu`, `silu`, `mish` (default: the backbone's own, GELU for CNN-3, ReLU for ResNet-18). An encoder trained with another is saved with its name in the file, and is loaded with the same flag. |
| `--max_rounds` | Cap of a classifier, in rounds (default 100). A round is as many training images as 100% of the labels, so 1% validates every 100 epochs and 10% every 10: every label fraction gets the same budget and the same patience. |
| `--patience` | Rounds with no better val accuracy before a classifier stops (default 5; the tuning uses 3). |
| `--tune` | A tuning run: chosen by val accuracy, so no test and no test curve; the folder has `tune` instead of `final`. |
| `--num_epochs` | Epochs of the DAE (`pretrain`), default `config.NUM_EPOCHS` (20). The classifiers have their own cap, `TARGET_NUM_EPOCHS`. |
| `--seed` | The training seed. The splits and the label subsets always use `config.SEED`. |
| `--search` | One trial of the hyperparameter search: proxy setting (10 epochs, the k-NN every 2 epochs with a bank of 10,000 images), no test set; the folder has `search` instead of `final`. |
| `--output_dir` | Where the logs, checkpoints and encoders go. |

Examples:

```bash
python train.py --mode finetune --encoder dae --label_fraction 0.1                # fine-tune with 10% of the labels
python train.py --mode linear_probe --encoder random --dataset cifar100           # random encoder, transfer
python train.py --mode pretrain --backbone resnet18 --decoder resnet18            # ResNet-18 encoder-decoder DAE
python train.py --mode linear_probe --encoder dae --backbone resnet18 --decoder resnet18   # its linear probe
```

## Data splits

- **train**: Tiny-ImageNet train minus the held-out images (90,000).
- **val**: 50 training images per class (10,000), for all tuning and model selection.
- **test**: the official Tiny-ImageNet validation set (10,000), only for the final numbers.

CIFAR-100 (the transfer set) is upscaled to 64×64 and split the same way: 45,000 train,
5,000 val, and its official test set (10,000). Both datasets are normalized with the mean
and standard deviation of Tiny-ImageNet.

## Final settings

The DAE, chosen by validation k-NN accuracy, not by reconstruction error:

| Setting | Value |
|---|---|
| Encoder | CNN-3: two 3×3 conv layers (32 and 64 channels, stride 2, GELU) and a Linear layer |
| h (`latent_dim`) | 256, the output of the encoder's last Linear layer |
| Corruption | salt and pepper, 50% of the pixels |
| Dropout | 0.4 on h, in training only |
| Optimizer | AdamW, lr 1e-3, weight decay 0.01 |
| Batch size | 512 |
| Augmentation, loss | horizontal flip, MSE |
| Epochs | 20; the kept checkpoint is the epoch with the best val k-NN |

The classifiers:

| Run | Settings |
|---|---|
| Linear probe | AdamW, lr 1e-3, no weight decay, batch 256, flip, label smoothing 0.1 (the defaults of `config.py`) |
| Fine-tuning and the supervised baseline | The recipe tuned on val (`config.TUNED`): SGD with Nesterov momentum, lr 0.003, weight decay 1e-5, dropout 0.5, `crop_jitter`, no label smoothing, batch 256. In fine-tuning the new head's learning rate is 3 times the encoder's. |

## Hyperparameter search

Every choice was made on val; the test set was never used. The searches were small grids
run one after the other, the winner of each carried to the next. Their launchers were
Colab notebooks that only looped over the calls below, and are not part of the repository.

- **The DAE, proxy setting** (221 trials): each trial trains on 10% of the training images
  for 10 epochs and is scored by val k-NN. The grids: the corruption and its level; the
  conv channels and h; dropout, augmentation and batch size; the optimizer and the
  activation; the learning rate, weight decay and scheduler; a second noise and combined
  augmentations; a smaller conv 2 and patch size.

  ```bash
  python train.py --mode pretrain --search --data_fraction 0.1 --noise_type patch_mask --noise_level 0.5
  ```

- **The DAE, all the training images** (47 trials): the leading settings for 20 epochs,
  with `--tune` (val only). The three best were then repeated with two more seeds (`--seed`).

  ```bash
  python train.py --mode pretrain --tune --num_epochs 20 --noise_type salt_pepper --noise_level 0.5 --dropout 0.4
  ```

- **The fine-tuning recipe** (33 runs): three grids one after the other (learning rate, the
  head's learning rate and batch size; dropout, augmentation and label smoothing; the
  optimizer and its weight decay), one hyperparameter at a time, 5 values, on 10% of the
  labels, by val accuracy.

  ```bash
  python train.py --mode finetune --encoder dae --tune --label_fraction 0.1 --patience 3 --max_rounds 50 --lr 3e-4
  ```

The logs of all of them are in the tuning folder (see Logs). The same folder also holds two
earlier, smaller searches (`search1` and `search2` in the folder names) and ResNet-18 runs.

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
| `colab_train_MA_A.ipynb` | Launcher: every final run of Part A. |
| `requirements.txt` | The packages, next to torch and torchvision. |

## Metrics and epochs

- Classifiers (`linear_probe`, `finetune`, `pretrain_supervise_encoder`) log top-1 and
  top-5 accuracy, and on val and test also macro precision, recall, F1 and AUROC
  (torchmetrics). After the test they save the confusion matrix (`confusion_matrix.csv`,
  `.png`) and the 20 most confused class pairs (`most_confused.csv`). `analysis` saves
  the same for the k-NN (`knn_*`).
- Every run that trains (not `--search` or `--tune`) also scores the test set after every
  epoch, with the same metrics, and k-NN for the encoders (`config.TEST_CURVE`). TensorBoard
  shows them as `epoch/test_*`, next to `epoch/train_*` and `epoch/val_*`. Only val
  chooses the checkpoint and stops a run, so the test curve is for plots only and
  the test numbers stay final numbers.
- Every classifier stops on `val_accuracy` after 5 rounds with no better one, with a cap
  of 100 rounds, and logs its calibration error (ECE) too. The DAE trains a fixed number of
  epochs (`--num_epochs`, default 20) and keeps the epoch with the best val k-NN.
- `finetune --encoder supervised --dataset cifar100 --label_fraction f` loads the
  supervised encoder trained on the same fraction f of the Tiny-ImageNet labels.

## Logs

The logs of every run of the report are shared on Google Drive (TensorBoard events, settings
and results):

- The 47 final runs: [midterm_final](https://drive.google.com/drive/folders/1r4T1nkcCEf8XEsXHHZT5ghV8tsOcA9ow?usp=sharing)
- The tuning runs: [midterm_runs](https://drive.google.com/drive/folders/1abQOyFbXNlbTOeBvXiSFEFRY8sn7_HuF?usp=sharing)

Every run writes to its own folder in `<output_dir>`:
`date<DDMMYYYY>_time<HHMMSS>_<model>_final_<stage>_<what>_seed<seed>`, for example
`date25112026_time123303_CNN3_final_analysis_DAE_on_TinyImageNet_seed42`. The date and
time are those of its first start; after a disconnect the run finds the same folder and
continues. The folder holds the TensorBoard logs, every setting (`hparams.yaml`), the
results (`results.json`) and `run_info.txt`: what the run is, its hyperparameters, epochs,
results, start and end time and training time (also in TensorBoard's TEXT tab). The
encoders go to `<output_dir>/encoders`. To view the logs:

```bash
tensorboard --logdir <output_dir>
```
