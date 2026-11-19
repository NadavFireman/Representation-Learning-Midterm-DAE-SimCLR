import os
import pandas as pd
import torch
import pytorch_lightning as pl
from datamodule import ImageDataModule
import config
from callbacks import MyPrintingCallback, EarlyStopping, ModelCheckpoint, ColabProgressBar, EpochMetricsCallback, KNNCallback
from pytorch_lightning.loggers import TensorBoardLogger
from utils import parse_args, build_model
from knn import knn_evaluate
torch.set_float32_matmul_precision("medium")  # to make lightning happy


def run(mode):
    """Train and evaluate one experiment. mode: config.MODEL_NAME, or --mode."""
    pl.seed_everything(config.TRAIN_SEED)  # same initialization and data order -> fair comparison

    # One folder per run, named by everything that defines it. The course named it
    # by the start time; a fixed name lets a run that was cut off (Colab disconnect)
    # find its own last.ckpt and continue. To train a run again from scratch,
    # delete its folder.
    encoder_name = os.path.basename(config.ENCODER_PATH)[:-3] if config.ENCODER_PATH else f"random_s{config.TRAIN_SEED}"
    run_name = f"{mode}_{encoder_name}"  # a search trial: encoder_name already ends with "_search"
    if mode != "pretrain":
        run_name += f"_{config.DATASET}_{config.LABEL_FRACTION}"
    if config.SEARCH and config.ENCODER_PATH is None:
        run_name += "_search"  # random-init encoder: no file name to carry it
    logger = TensorBoardLogger(os.path.join(config.OUTPUT_DIR, "tb_logs"), name="", version=run_name, default_hp_metric=False)
    model = build_model(mode)

    # The DAE uses every training image and no labels. The other modes use the
    # class-balanced labeled part (--label_fraction). The dataset is --dataset, for every mode.
    label_fraction = 1.0 if mode == "pretrain" else config.LABEL_FRACTION
    if config.SEARCH:
        num_epochs = config.SWEEP_NUM_EPOCHS  # proxy setting for tuning; full setting for final models only
    elif mode in config.PRETRAIN_MODES:
        num_epochs = config.NUM_EPOCHS
    else:
        num_epochs = config.TARGET_NUM_EPOCHS

    # This run's own config file: every value of config.py as this run used it
    # (after the command-line overrides). Lightning's log_hyperparams writes it to
    # hparams.yaml next to the logs, and to the HPARAMS tab of TensorBoard.
    settings = {name: getattr(config, name) for name in dir(config) if name.isupper()}
    settings.update(MODE=mode, RUN_NUM_EPOCHS=num_epochs, RUN_LABEL_FRACTION=label_fraction)
    logger.log_hyperparams(settings)

    dm = ImageDataModule(
        data_dir=config.DATA_DIR,
        batch_size=config.BATCH_SIZE,
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
        return results

    callbacks = [MyPrintingCallback(), ColabProgressBar(refresh_rate=20), EpochMetricsCallback()]
    if mode in config.PRETRAIN_MODES:
        callbacks.append(KNNCallback())  # val k-NN every epoch, test k-NN at the end
    # save_last=True also writes last.ckpt every epoch, to continue after a disconnect.
    checkpoint_dir = os.path.join(logger.log_dir, "checkpoints")
    if mode == "pretrain":
        # Fixed number of epochs, no early stopping. The DAE is judged by its h,
        # not by reconstruction, so the kept epoch is the one with the best val k-NN.
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_knn_top1", mode="max", save_last=True))
    else:
        callbacks.append(EarlyStopping(monitor="val_loss"))
        callbacks.append(ModelCheckpoint(dirpath=checkpoint_dir, monitor="val_loss", mode="min", save_last=True))

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
    print(model)
    details = (
        f"Training mode is: {mode}, on {config.DATASET} for {num_epochs} epochs, "
        f"batch size {config.BATCH_SIZE}, label fraction {label_fraction}, latent_dim {config.LATENT_DIM}, "
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
    return results[0]


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
        elif args.mode == "knn":
            print(f"  {args.mode:<28} ({config.DATASET}): k-NN top-1 {results['test_knn_top1']:.3f}, "
                  f"top-5 {results['test_knn_top5']:.3f}")
        else:
            print(f"  {args.mode:<28} ({config.DATASET}): accuracy {results['test_accuracy']:.3f}, "
                  f"top-5 {results['test_top5_accuracy']:.3f}")
