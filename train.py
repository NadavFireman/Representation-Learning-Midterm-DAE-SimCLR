import json
import os
import pandas as pd
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer
from pytorch_lightning.loggers import TensorBoardLogger
from datamodule import ImageDataModule
import config
from callbacks import (MyPrintingCallback, EarlyStopping, ModelCheckpoint, ColabProgressBar, EpochMetricsCallback,
                       KNNCallback, ConfusionMatrixCallback)
from utils import parse_args, build_model
from knn import knn_evaluate
from analysis import run_analysis
import run_info
torch.set_float32_matmul_precision("medium")  # to make lightning happy


def run(mode):
    """Train and evaluate one experiment. mode: config.MODEL_NAME, or --mode."""
    pl.seed_everything(config.TRAIN_SEED)  # same initialization and data order -> fair comparison

    # One folder per run; after a Colab disconnect the run finds it and continues from last.ckpt
    run_name = run_info.folder_name(mode)
    logger = TensorBoardLogger(config.OUTPUT_DIR, name="", version=run_name, default_hp_metric=False)

    # A finished run has results.json: running it again prints its results and stops.
    results_file = os.path.join(logger.log_dir, "results.json")
    if os.path.exists(results_file):
        with open(results_file) as f:
            results = json.load(f)
        print(f"{run_name} already finished; its results are in {results_file}")
        return results
    start = run_info.now()
    done = {}  # the trainer and the training time, for run_info.txt

    def save_results(results):
        """Write the final numbers to results.json and run_info.txt, and return them."""
        results = {key: float(value) for key, value in results.items()}  # tensors and numpy numbers -> float
        os.makedirs(logger.log_dir, exist_ok=True)
        with open(results_file, "w") as f:
            json.dump(results, f, indent=1)
        run_info.write(logger, mode, model, results, start, done.get("trainer"), done.get("train_seconds"))
        return results

    model = build_model(mode)

    # The DAE trains on --data_fraction of the images, the other modes on --label_fraction
    label_fraction = config.DATA_FRACTION if mode == "pretrain" else config.LABEL_FRACTION
    if config.SEARCH:
        num_epochs = config.SWEEP_NUM_EPOCHS  # proxy setting for tuning
    elif mode == "pretrain":
        num_epochs = config.NUM_EPOCHS  # the DAE: fixed, no early stopping
    else:
        num_epochs = config.TARGET_NUM_EPOCHS  # the classifiers: a cap in rounds, EarlyStopping ends them sooner
    # Epochs per round: a round is as many images as 100% of the labels (config.TARGET_NUM_EPOCHS)
    every = 1 if mode == "pretrain" or config.SEARCH else max(1, round(1 / label_fraction))
    # A DAE search trial scores its k-NN every SEARCH_KNN_EVERY epochs only, to be faster
    val_every = config.SEARCH_KNN_EVERY if mode == "pretrain" and config.SEARCH else every

    # This run's own config file: every value of config.py as this run used it (hparams.yaml).
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
        test_curve=config.TEST_CURVE and not config.SEARCH,  # a search trial never uses the test set
        eval_batch_size=config.EVAL_BATCH_SIZE,
        augment=config.DAE_AUGMENT if mode == "pretrain" else config.AUGMENT,
    )

    if mode == "knn":
        # No training: k-NN accuracy of the saved (or random) encoder.
        dm.prepare_data()
        dm.setup("test")
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        splits = ("val",) if config.SEARCH else ("val", "test")
        results = {f"{split}_knn_{k}": v for split, (top1, top5) in knn_evaluate(model.encoder, dm, config.KNN_K, splits).items()
                   for k, v in (("top1", top1), ("top5", top5))}
        logger.log_metrics(results)
        logger.save()  # writes hparams.yaml
        return results if config.SEARCH else save_results(results)

    if mode == "analysis":
        # No training: the A5 evidence and the robustness test.
        return save_results(run_analysis(model.encoder, dm, logger))

    timer = Timer()  # the training time; kept in the checkpoint, so it adds up over a disconnect
    callbacks = [MyPrintingCallback(), ColabProgressBar(refresh_rate=20), EpochMetricsCallback(), timer]
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    if mode == "pretrain":
        callbacks.append(KNNCallback())  # val k-NN (and the test curve) every epoch, test k-NN at the end
        # The DAE is judged by its h: keep the epoch with the best val k-NN.
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_knn_top1", mode="max"))
    else:
        callbacks.append(ConfusionMatrixCallback())  # a classifier: its test confusion matrix
        # Stop on and keep the best val_accuracy.
        callbacks.append(EarlyStopping(monitor="val_accuracy", mode="max", patience=config.PATIENCE))
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_accuracy", mode="max"))
    # last.ckpt, to continue after a disconnect. Second, so ckpt_path="best" is the monitored one
    callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, filename="last", enable_version_counter=False,
                                     every_n_epochs=every))  # once per round, not every short epoch

    trainer = pl.Trainer(
        logger=logger,
        accelerator=config.ACCELERATOR,
        devices=config.DEVICES,
        min_epochs=1,
        max_epochs=num_epochs * every,
        check_val_every_n_epoch=val_every,
        precision=config.PRECISION,
        benchmark=True,  # cuDNN picks the fastest convolutions for the fixed 64x64 input
        limit_train_batches=config.LIMIT_BATCHES,
        limit_val_batches=config.LIMIT_BATCHES,
        limit_test_batches=config.LIMIT_BATCHES,
        log_every_n_steps=10,  # default is 50: short runs would get no step points
        callbacks=callbacks,
    )
    done["trainer"] = trainer

    print(model)
    print(f"Training mode is: {mode}, on {config.DATASET} for up to {num_epochs} rounds of {every} epochs, batch size {batch_size}, "
          f"label fraction {label_fraction}, seed {config.TRAIN_SEED}, run {run_name}")

    # ckpt_path="last": continue from this run's last.ckpt if it exists.
    trainer.fit(model, dm, ckpt_path="last")
    done["train_seconds"] = timer.time_elapsed()  # now: validate and test below reload the Timer's state
    # Validate the kept checkpoint (this loads its weights). The test is left to trainer.test
    dm.test_curve = False
    val_results = trainer.validate(model, dm, ckpt_path="best")

    if mode in config.SAVE_ENCODER_MODES:
        # Save only the encoder.
        os.makedirs(os.path.dirname(config.ENCODER_PATH), exist_ok=True)
        torch.save(model.encoder.state_dict(), config.ENCODER_PATH)

    if config.TUNE:
        # A tuning run: chosen by val, so the test set is not touched.
        return save_results({k: v for k, v in val_results[0].items() if k.startswith("val_")})

    if config.SEARCH:
        # A search trial: no test. A DAE trial adds one row to sweep_results.csv.
        if mode == "pretrain":
            row = {name: getattr(config, name.upper()) for name in config.SWEEP}  # "noise_type" -> config.NOISE_TYPE
            row["seed"] = config.TRAIN_SEED
            row.update({key: val_results[0][key] for key in ("val_loss", "val_knn_top1", "val_knn_top5")})
            sweep_results = os.path.join(config.OUTPUT_DIR, "sweep_results.csv")
            rows = pd.DataFrame([row])
            if os.path.exists(sweep_results):
                rows = pd.concat([pd.read_csv(sweep_results), rows])
            # One row per trial: a trial that is run again replaces its row.
            rows.drop_duplicates(subset=[*config.SWEEP, "seed"], keep="last").to_csv(sweep_results, index=False)
        return save_results(val_results[0])  # results.json: a finished trial is skipped when run again

    results = trainer.test(model, dm, ckpt_path="best")
    # The val numbers of the kept checkpoint too, next to the test numbers
    return save_results({**{k: v for k, v in val_results[0].items() if k.startswith("val_")}, **results[0]})


if __name__ == "__main__":
    args = parse_args()
    results = run(args.mode)

    if config.SEARCH or config.TUNE:
        print("\nValidation results (search or tuning run):")
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
