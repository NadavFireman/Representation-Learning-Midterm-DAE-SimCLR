import json
import os
import shutil
import pandas as pd
import torch
import pytorch_lightning as pl
from datamodule import ImageDataModule
import config
from callbacks import (MyPrintingCallback, EarlyStopping, ModelCheckpoint, ColabProgressBar, EpochMetricsCallback,
                       KNNCallback, ConfusionMatrixCallback)
import yaml
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_args, build_model, architecture_tag, protocol_tag
from knn import knn_evaluate
from analysis import run_analysis
torch.set_float32_matmul_precision("medium")  # to make lightning happy

# The modes --retest works on: the classifiers (tested again from their best
# checkpoint) and analysis (no checkpoint, it runs again).
RETEST_MODES = ("linear_probe", "finetune", "pretrain_supervise_encoder", "analysis")
# How much a number the run already reported may move in a retest: GPU
# nondeterminism changes the last digits, a real change in the evaluation more.
RETEST_TOLERANCE = 0.0005


def best_checkpoint(checkpoint_dir):
    """The checkpoint file a finished run kept as its best.

    trainer.test(ckpt_path="best") only knows it in the process that ran fit.
    Every checkpoint also stores the state of each callback, and the monitored
    ModelCheckpoint's state has best_model_path, so last.ckpt knows it too. That
    path is the one of the machine that trained (Colab), so only its file name is
    used, inside this run's checkpoints folder.
    """
    last = torch.load(os.path.join(checkpoint_dir, "last.ckpt"), map_location="cpu", weights_only=False)
    for name, state in last["callbacks"].items():
        if name.startswith("ModelCheckpoint") and state.get("monitor"):  # the last.ckpt callback has no monitor
            return os.path.join(checkpoint_dir, os.path.basename(state["best_model_path"]))
    raise FileNotFoundError(f"no monitored ModelCheckpoint in {checkpoint_dir}/last.ckpt")


def run(mode):
    """Train and evaluate one experiment. mode: config.MODEL_NAME, or --mode."""
    pl.seed_everything(config.TRAIN_SEED)  # same initialization and data order -> fair comparison

    # One folder per run, named by everything that defines it. The course named it
    # by the start time; a fixed name lets a run that was cut off (Colab disconnect)
    # find its own last.ckpt and continue. To train a run again from scratch,
    # move its folder out of tb_logs.
    random_name = f"random{architecture_tag('random')}_s{config.TRAIN_SEED}"  # random_s42, random_resnet18_s42
    encoder_name = os.path.basename(config.ENCODER_PATH)[:-3] if config.ENCODER_PATH else random_name
    run_name = f"{mode}_{encoder_name}"  # a search trial: encoder_name already ends with "_search"
    if mode != "pretrain":
        run_name += f"_{config.DATASET}_{config.LABEL_FRACTION}"
    if config.SEARCH and config.ENCODER_PATH is None:
        run_name += "_search"  # random-init encoder: no file name to carry it
    if mode in ("linear_probe", "finetune"):
        run_name += protocol_tag()  # "_acc": the checkpoint is chosen by val_accuracy (config.PROTOCOL)
    # pretrain_supervise_encoder already has it: its name comes from its encoder file (utils.encoder_path).
    logger = TensorBoardLogger(os.path.join(config.OUTPUT_DIR, "tb_logs"), name="", version=run_name, default_hp_metric=False)

    # A run that finished writes its results to results.json in its folder (see
    # save_results). Running it again (a loop restarted after a disconnect) prints
    # those results and stops. Resuming from last.ckpt alone is not enough: Lightning
    # does not save that EarlyStopping stopped the run, so an early-stopped run
    # would train more epochs and could replace its test results.
    # Search trials always run again: train.py rewrites their row in sweep_results.csv.
    # --retest (config.RETEST) is the exception: a finished classifier run is tested
    # again from its best checkpoint, with no training, to add metrics that were
    # added to the code after it ran (see the end of this function). analysis has
    # no checkpoint: it simply runs again.
    results_file = os.path.join(logger.log_dir, "results.json")
    retest = config.RETEST and mode in RETEST_MODES and os.path.exists(results_file)
    if config.RETEST and mode in RETEST_MODES and mode != "analysis" and not retest:
        # --retest never trains. No results.json here means a wrong flag or --output_dir,
        # or a run that has not finished: stop, instead of starting a training run.
        raise FileNotFoundError(f"--retest: {run_name} has no results.json in {logger.log_dir}; nothing to test")
    if os.path.exists(results_file) and not config.SEARCH and not retest:
        with open(results_file) as f:
            results = json.load(f)
        print(f"{run_name} already finished; its results are in {results_file}")
        return results

    def save_results(results):
        """Write the final numbers of this run to results.json, and return them."""
        results = {key: float(value) for key, value in results.items()}  # tensors and numpy numbers -> float
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=1)
        return results

    def save_retest_results(new_results, key):
        """--retest: add the new metrics to results.json, and keep what the run reported.

        The first results.json is kept as results_before_retest.json (once; a second
        retest keeps that same copy). The number the run already had (key: test_accuracy,
        or test_knn_top1 for analysis) must come out the same, or nothing is written:
        a change there means the evaluation changed, not just new metrics.
        """
        backup = os.path.join(logger.log_dir, "results_before_retest.json")
        if not os.path.exists(backup):
            shutil.copyfile(results_file, backup)
        with open(backup) as f:
            old_results = json.load(f)
        new_results = {k: float(v) for k, v in new_results.items()}
        if abs(new_results[key] - old_results[key]) > RETEST_TOLERANCE:
            raise RuntimeError(f"--retest: {run_name} {key} was {old_results[key]:.4f}, now {new_results[key]:.4f}. "
                               f"results.json is left as it was.")
        return save_results({**old_results, **new_results})

    model = build_model(mode)

    # The DAE uses every training image and no labels. The other modes use the
    # class-balanced labeled part (--label_fraction). The dataset is --dataset, for every mode.
    label_fraction = 1.0 if mode == "pretrain" else config.LABEL_FRACTION
    if config.SEARCH:
        num_epochs = config.SWEEP_NUM_EPOCHS  # proxy setting for tuning; full setting for final models only
    elif mode == "pretrain":
        num_epochs = config.NUM_EPOCHS  # the DAE: a fixed number, its checkpoint is chosen by val k-NN
    else:
        # Every classifier, the supervised encoder too: TARGET_NUM_EPOCHS is only a cap,
        # EarlyStopping on val_accuracy ends the run sooner.
        num_epochs = config.TARGET_NUM_EPOCHS

    # This run's own config file: every value of config.py as this run used it
    # (after the command-line overrides). Lightning's log_hyperparams writes it to
    # hparams.yaml next to the logs, and to the HPARAMS tab of TensorBoard.
    # A retest keeps the file of the run it tests: those are the settings it trained with.
    hparams_file = os.path.join(logger.log_dir, "hparams.yaml")
    if retest and os.path.exists(hparams_file):
        # utf-8: the paths in it can be Hebrew, and Windows would read it as cp1252.
        # full_load, as Lightning's own reader: the file holds tuples (PRETRAIN_MODES ...).
        with open(hparams_file, encoding="utf-8") as f:
            settings = yaml.full_load(f)
    else:
        settings = {name: getattr(config, name) for name in dir(config) if name.isupper()}
        settings.update(MODE=mode, RUN_NUM_EPOCHS=num_epochs, RUN_LABEL_FRACTION=label_fraction)
    logger.log_hyperparams(settings)

    batch_size = config.DAE_BATCH_SIZE if mode == "pretrain" else config.BATCH_SIZE
    dm = ImageDataModule(
        data_dir=config.DATA_DIR,
        batch_size=batch_size,
        num_workers=config.NUM_WORKERS,
        dataset_name=config.DATASET,
        label_fraction=label_fraction,
        seed=config.SEED,
    )

    if mode == "knn":
        # No training: k-NN accuracy of the saved (or random) encoder. The memory
        # bank is always every training image (see datamodule.bank_dataloader).
        dm.prepare_data()
        dm.setup("test")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        val_top1, val_top5 = knn_evaluate(model.encoder, dm, config.KNN_K, "val")
        print(f"Training mode is: knn, encoder {encoder_name}, on {config.DATASET}")
        results = {"val_knn_top1": val_top1, "val_knn_top5": val_top5}
        if not config.SEARCH:
            results["test_knn_top1"], results["test_knn_top5"] = knn_evaluate(model.encoder, dm, config.KNN_K, "test")
        logger.log_metrics(results)
        logger.save()  # writes hparams.yaml (no Trainer here to do it)
        return results if config.SEARCH else save_results(results)

    if mode == "analysis":
        # No training: the A5 evidence and the robustness test of the saved (or random) encoder.
        results = run_analysis(model.encoder, dm, logger)
        return save_retest_results(results, "test_knn_top1") if retest else save_results(results)

    callbacks = [MyPrintingCallback(), ColabProgressBar(refresh_rate=20), EpochMetricsCallback()]
    if mode in config.PRETRAIN_MODES:
        callbacks.append(KNNCallback())  # val k-NN every epoch, test k-NN at the end
    if mode != "pretrain":
        callbacks.append(ConfusionMatrixCallback())  # a classifier: its test confusion matrix
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    if mode == "pretrain":
        # Fixed number of epochs, no early stopping. The DAE is judged by its h,
        # not by reconstruction, so the kept epoch is the one with the best val k-NN.
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_knn_top1", mode="max"))
    else:
        # config.PROTOCOL: "acc" stops on and keeps the best val_accuracy, "loss" the lowest val_loss.
        if config.PROTOCOL == "acc":
            callbacks.append(EarlyStopping(monitor="val_accuracy", mode="max", patience=config.PATIENCE))
            callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_accuracy", mode="max"))
        else:
            callbacks.append(EarlyStopping(monitor="val_loss"))
            callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min"))
    # last.ckpt, rewritten every epoch, to continue after a disconnect. (A run that
    # finished is caught earlier, by its results.json.) save_last=True on the callback above
    # would rewrite it only when the monitored metric improves, so a finished run
    # would train again from its best epoch. This callback comes second, so
    # ckpt_path="best" below still means the monitored one.
    callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, filename="last", enable_version_counter=False))

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=num_epochs,
        precision=config.PRECISION,
        limit_train_batches=config.LIMIT_BATCHES,
        limit_val_batches=config.LIMIT_BATCHES,
        limit_test_batches=config.LIMIT_BATCHES,
        log_every_n_steps=10,  # default is 50: short runs would get no step points
        callbacks=callbacks,
    )
    if retest:
        # No fit: test the checkpoint this run kept as its best, and add the new
        # metrics to its results.json (save_retest_results checks the old numbers).
        results = trainer.test(model, dm, ckpt_path=best_checkpoint(checkpoint_dir))
        return save_retest_results(results[0], "test_accuracy")

    print(model)
    details = (
        f"Training mode is: {mode}, on {config.DATASET} for {num_epochs} epochs, "
        f"batch size {batch_size}, label fraction {label_fraction}, latent_dim {config.LATENT_DIM}, "
        f"seed {config.TRAIN_SEED}"
    )
    if mode == "pretrain":
        details += f", noise {config.NOISE_TYPE} {config.NOISE_LEVEL}"
    else:
        details += f", encoder {encoder_name}"
    if config.SEARCH:
        details += " (search trial: proxy setting, no test)"
    print(details)

    # ckpt_path="last": Lightning continues from this run's last.ckpt if it exists
    # (after a disconnect), and starts from scratch if it does not.
    trainer.fit(model, dm, ckpt_path="last")
    # Validate (and test) the kept checkpoint, not the last epoch. This also loads
    # its weights into model, so the encoder saved below is the kept one.
    val_results = trainer.validate(model, dm, ckpt_path="best")

    if mode in config.SAVE_ENCODER_MODES:
        # Keep the encoder. The decoder or the classification head is not saved.
        os.makedirs(os.path.dirname(config.ENCODER_PATH), exist_ok=True)
        torch.save(model.encoder.state_dict(), config.ENCODER_PATH)

    if config.SEARCH:
        # A search trial: the test set is not touched (only for final numbers).
        # A DAE trial adds one row to sweep_results.csv: its hyperparameters and
        # the val metrics of the kept checkpoint.
        if mode == "pretrain":
            row = {name: getattr(config, name.upper()) for name in config.SWEEP}  # "noise_type" -> config.NOISE_TYPE
            row["seed"] = config.TRAIN_SEED
            row.update({key: val_results[0][key] for key in ("val_loss", "val_knn_top1", "val_knn_top5")})
            sweep_results = os.path.join(config.OUTPUT_DIR, "sweep_results.csv")
            rows = pd.DataFrame([row])
            if os.path.exists(sweep_results):
                rows = pd.concat([pd.read_csv(sweep_results), rows])
            # A trial that is run again (it only reloads last.ckpt) replaces its old row: one row per trial.
            rows.drop_duplicates(subset=[*config.SWEEP, "seed"], keep="last").to_csv(sweep_results, index=False)
        return val_results[0]

    results = trainer.test(model, dm, ckpt_path="best")
    return save_results(results[0])


if __name__ == "__main__":
    args = parse_args()
    results = run(args.mode)

    if config.SEARCH:
        print("\nValidation results (search trial):")
        print("  " + ", ".join(f"{key} {value:.4f}" for key, value in results.items()))
    else:
        print("\nTest results:")
        if args.mode == "pretrain":
            print(f"  {args.mode:<28} ({config.DATASET}): reconstruction MSE {results['test_loss']:.4f}, "
                  f"k-NN top-1 {results['test_knn_top1']:.3f}, top-5 {results['test_knn_top5']:.3f}")
        elif args.mode in ("knn", "analysis"):
            print(f"  {args.mode:<28} ({config.DATASET}): k-NN top-1 {results['test_knn_top1']:.3f}, "
                  f"top-5 {results['test_knn_top5']:.3f}")
        else:
            print(f"  {args.mode:<28} ({config.DATASET}): accuracy {results['test_accuracy']:.3f}, "
                  f"top-5 {results['test_top5_accuracy']:.3f}")
