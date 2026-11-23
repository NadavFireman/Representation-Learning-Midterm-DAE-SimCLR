"""The run folder name, date<DDMMYYYY>_time<HHMMSS>_<model>_<tag>_<stage>_<what>_<hyperparameters>_seed<seed>, and run_info.txt.

The date and time are those of the run's first start. A run that continues after a
disconnect finds its folder by the part after them.
"""
import os
import re
from datetime import datetime
from zoneinfo import ZoneInfo

import config

TZ = ZoneInfo("Asia/Jerusalem")
STAGE = {"pretrain": "pretrain", "pretrain_supervise_encoder": "supervised", "knn": "knn",
         "linear_probe": "linear_probe", "finetune": "finetune", "analysis": "analysis"}
PERCENT = {0.01: "1pct", 0.1: "10pct", 1.0: "100pct"}
# The DAE settings as config.py has them, before the flags change them: a name lists the ones that differ
DAE_ARCHITECTURE = ("CHANNELS", "ACTIVATION", "PATCH_SIZE")
DAE_TRAINING = ("LEARNING_RATE", "DAE_BATCH_SIZE", "WEIGHT_DECAY", "DAE_OPTIMIZER", "DAE_DROPOUT", "DAE_AUGMENT",
                "DAE_SCHEDULER", "DATA_FRACTION", "NUM_EPOCHS")
DAE_DEFAULTS = {name: getattr(config, name) for name in ("LATENT_DIM", *DAE_ARCHITECTURE, *DAE_TRAINING)}
DAE_TAGS = {"LATENT_DIM": "h", "CHANNELS": "c", "ACTIVATION": "act-", "PATCH_SIZE": "p", "LEARNING_RATE": "lr",
            "DAE_BATCH_SIZE": "bs", "WEIGHT_DECAY": "wd", "DAE_OPTIMIZER": "opt-", "DAE_DROPOUT": "do",
            "DAE_AUGMENT": "aug-", "DAE_SCHEDULER": "sched-", "DATA_FRACTION": "data", "NUM_EPOCHS": "e"}


def now():
    return datetime.now(TZ)


def dae_tags(names):
    """The DAE settings among names that differ from config.py, as a name part, e.g. "_c16-32_lr0.002"."""
    text = ""
    for name in names:
        value = getattr(config, name)
        if value == DAE_DEFAULTS[name] or (name == "PATCH_SIZE" and config.NOISE_TYPE != "patch_mask"):
            continue  # the patch size matters only for patch_mask
        if isinstance(value, tuple):
            value = "-".join(map(str, value))
        text += f"_{DAE_TAGS[name]}{value:g}" if isinstance(value, float) else f"_{DAE_TAGS[name]}{value}"
    return text


def what(mode):
    """The encoder (DAE, AE, random, supervised), and the dataset and labels of the run."""
    name = os.path.basename(config.ENCODER_PATH or "random")
    if name.startswith("dae_"):
        encoder = "AE" if config.NOISE_LEVEL == 0 else "DAE"
        if config.NOISE_LEVEL != 0 and (config.NOISE_TYPE, config.NOISE_LEVEL) != ("salt_pepper", 0.5):
            encoder += f"-{config.NOISE_TYPE}{config.NOISE_LEVEL}"
        if config.BACKBONE == "resnet18" and config.DECODER == "light":
            encoder += "-lightdec"
    else:
        encoder = "supervised" if name.startswith("supervised") else "random"
    if mode == "pretrain_supervise_encoder":
        return PERCENT[config.LABEL_FRACTION]
    if mode == "pretrain":
        return encoder
    text = f"{encoder}_on_{config.DATASET}"
    return text + f"_{PERCENT[config.LABEL_FRACTION]}" if mode == "finetune" else text


def folder_name(mode):
    """The existing folder of this run (it continues), or a new name with the date and time now."""
    model = "ResNet18" if config.BACKBONE == "resnet18" else "CNN3"
    stage = STAGE[mode]
    if config.FINETUNE_FROM:
        stage += f"-from-{config.FINETUNE_FROM}"
    suffix = f"{model}_{config.RUN_TAG}_{stage}_{what(mode)}"
    if mode == "pretrain":  # a search or tuning trial also lists its training settings
        trial = config.SEARCH or config.TUNE
        suffix += dae_tags(("LATENT_DIM", *DAE_ARCHITECTURE, *(DAE_TRAINING if trial else ())))
    if mode in ("linear_probe", "finetune", "pretrain_supervise_encoder"):  # the hyperparameters that differ from config
        for tag, value, default in (("lr", config.CLASSIFIER_LR, None), ("hm", config.HEAD_LR_MULT, 1.0),
                                    ("bs", config.BATCH_SIZE, 256), ("wd", config.CLASSIFIER_WD, 0.0),
                                    ("do", config.DROPOUT, 0.0), ("aug-", config.AUGMENT, "flip"),
                                    ("ls", config.LABEL_SMOOTHING, 0.1), ("opt-", config.OPTIMIZER, "adamw"),
                                    ("act-", config.ACTIVATION, None)):
            if value != default:
                suffix += f"_{tag}{value:g}" if isinstance(value, float) else f"_{tag}{value}"
    if mode in ("knn", "analysis"):
        suffix += dae_tags(("CHANNELS", "ACTIVATION"))  # the encoder they load
    elif mode != "pretrain":
        suffix += dae_tags(("CHANNELS",))
    suffix += f"_seed{config.TRAIN_SEED}"  # every run, seed 42 too
    if os.path.isdir(config.OUTPUT_DIR):
        for name in os.listdir(config.OUTPUT_DIR):
            if re.fullmatch(r"date\d{8}_time\d{6}_" + re.escape(suffix), name):
                return name
    return now().strftime("date%d%m%Y_time%H%M%S_") + suffix


def settings(mode, model):
    """The hyperparameters of the run, in one line (all of them are in hparams.yaml)."""
    parts = [f"encoder {'ResNet-18' if config.BACKBONE == 'resnet18' else 'CNN-3'}", f"h {config.LATENT_DIM}"]
    if mode == "pretrain":
        noise = f"{config.NOISE_TYPE} {config.NOISE_LEVEL}" + (f" (patch {config.PATCH_SIZE})" if config.NOISE_TYPE == "patch_mask" else "")
        epochs = config.SWEEP_NUM_EPOCHS if config.SEARCH else config.NUM_EPOCHS
        parts += [f"channels {config.CHANNELS[0]}-{config.CHANNELS[1]}", f"activation {config.ACTIVATION or 'the backbone own'}",
                  f"decoder {config.DECODER}", f"noise {noise}", f"dropout on h {config.DAE_DROPOUT}",
                  f"augment {config.DAE_AUGMENT}", f"{config.DAE_OPTIMIZER} lr {config.LEARNING_RATE}, weight decay {config.WEIGHT_DECAY}",
                  f"scheduler {config.DAE_SCHEDULER}", f"batch {config.DAE_BATCH_SIZE}",
                  f"{config.DATA_FRACTION:g} of the training images", f"{epochs} epochs, checkpoint: best val k-NN"]
    elif mode in ("knn", "analysis"):
        parts += [f"k-NN k={config.KNN_K}, no training"]
    else:
        trains = "head only" if model.hparams.freeze_encoder else f"from {config.FINETUNE_FROM}" if config.FINETUNE_FROM else "all weights"
        parts += [f"trains {trains}", f"{PERCENT[config.LABEL_FRACTION]} of the labels",
                  f"{model.hparams.optimizer} lr {model.hparams.learning_rate} (head x{model.hparams.head_lr_mult}), weight decay {model.hparams.weight_decay}",
                  f"activation {model.hparams.activation or 'the backbone own'}",
                  f"batch {config.BATCH_SIZE}", f"augment {config.AUGMENT}", f"dropout {model.dropout.p}",
                  f"label smoothing {config.LABEL_SMOOTHING}", f"up to {config.TARGET_NUM_EPOCHS} rounds, patience {config.PATIENCE}"]
    if config.ENCODER_PATH and mode not in config.SAVE_ENCODER_MODES:
        parts.append(f"encoder file {os.path.basename(config.ENCODER_PATH)}")
    return ", ".join(parts + [f"dataset {config.DATASET}", f"seed {config.TRAIN_SEED}"])


def write(logger, mode, model, results, start, trainer=None, timer=None):
    """run_info.txt in the run folder, and the same in TensorBoard's TEXT tab."""
    name = os.path.basename(logger.log_dir)
    started = datetime.strptime(name[4:23], "%d%m%Y_time%H%M%S").replace(tzinfo=TZ)
    minutes = (timer.time_elapsed() if timer else (now() - start).total_seconds()) / 60
    epochs = "no training"
    if trainer is not None:
        best = [c for c in trainer.checkpoint_callbacks if c.monitor][0]
        every = trainer.check_val_every_n_epoch  # epochs per round (1 for the DAE and 100% of the labels)
        unit = "epochs" if every == 1 else f"rounds of {every} epochs"
        epochs = f"{trainer.current_epoch // every} {unit} trained"
        if best.best_model_score is not None:
            epoch = int(re.search(r"epoch=(\d+)", best.best_model_path).group(1))
            epochs += f", best is {epoch // every + 1} ({best.monitor} {float(best.best_model_score):.4f})"
    lines = [("what", f"{mode}: {name[24:]}"), ("hyperparameters", settings(mode, model)), ("epochs", epochs),
             ("results", ", ".join(f"{k} {v:.4f}" for k, v in results.items() if k.startswith(("test_", "val_")))),
             ("started", f"{started:%Y-%m-%d %H:%M:%S} (Israel time)"), ("finished", f"{now():%Y-%m-%d %H:%M:%S}"),
             ("training time", f"{minutes:.1f} minutes"), ("folder", name)]
    with open(os.path.join(logger.log_dir, "run_info.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(f"{k}: {v}" for k, v in lines) + "\n")
    logger.experiment.add_text("run_info", "| | |\n|---|---|\n" + "\n".join(f"| **{k}** | {v} |" for k, v in lines))
    logger.experiment.flush()
    print(f"[{now():%Y-%m-%d %H:%M:%S}] {name} finished, {minutes:.1f} minutes")
