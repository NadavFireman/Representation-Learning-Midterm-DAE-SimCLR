from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, Callback, TQDMProgressBar
from pytorch_lightning.callbacks.progress.tqdm_progress import Tqdm

from torchmetrics.functional import accuracy

import config
from knn import extract_features, knn_scores


class ColabProgressBar(TQDMProgressBar):
    """tqdm bar without the separate validation bar, which Colab prints as a new line per update."""

    def init_validation_tqdm(self):
        return Tqdm(disable=True)


class MyPrintingCallback(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


class EpochMetricsCallback(Callback):
    """Logs one point per validation under "epoch/...", with the round as the x-axis, so runs
    with different numbers of steps line up. The test curve goes under "epoch/test_..."."""

    METRICS = ["loss", "accuracy", "top5_accuracy", "precision", "recall", "f1", "auroc", "calibration_error",
               "knn_top1", "knn_top5"]

    def _log_metrics(self, trainer, names):
        metrics = {}
        for name in names:
            # with on_step=True Lightning stores the epoch value as "<name>_epoch"
            value = trainer.callback_metrics.get(f"{name}_epoch", trainer.callback_metrics.get(name))
            if value is not None:
                metrics[f"epoch/{name.replace('testcurve_', 'test_')}"] = value.item()

        if metrics and trainer.logger is not None:
            trainer.logger.log_metrics(metrics, step=trainer.current_epoch // trainer.check_val_every_n_epoch)

    def on_train_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % trainer.check_val_every_n_epoch == 0:  # once per round
            self._log_metrics(trainer, ["train_loss", "train_accuracy"])

    def on_validation_end(self, trainer, pl_module):
        # Skip the sanity check and the extra trainer.validate() after training
        if trainer.sanity_checking or trainer.state.fn != "fit":
            return
        self._log_metrics(trainer, [f"{split}_{m}" for split in ("val", "testcurve") for m in self.METRICS])


class ConfusionMatrixCallback(Callback):
    """After the test of a classifier: saves its test confusion matrix (analysis.save_confusion_matrix)."""

    def on_test_epoch_end(self, trainer, pl_module):
        # Imported here: analysis.py switches matplotlib to files only (Agg)
        from analysis import save_confusion_matrix

        confusion = pl_module.test_confusion.compute()  # [num_classes, num_classes]
        pl_module.test_confusion.reset()
        save_confusion_matrix(confusion, trainer.datamodule.class_names(), trainer.logger)


class KNNCallback(Callback):
    """k-NN accuracy of the encoder's h: val_knn_* after every validation, test_knn_* after the test.
    ModelCheckpoint watches val_knn_top1, so the DAE is selected by k-NN, not by reconstruction.
    On the kept checkpoint also the macro precision, recall, F1 and AUROC of the vote."""

    def __init__(self, k=None, bank_size=None):
        super().__init__()
        self.k = k if k is not None else config.KNN_K
        if bank_size is None:
            # A search trial uses the smaller bank; final runs use all of train.
            bank_size = config.SEARCH_KNN_BANK_SIZE if config.SEARCH else config.KNN_BANK_SIZE
        self.bank_size = bank_size

    def _log_knn(self, trainer, pl_module, splits):
        # One bank for all the splits. The per-epoch test k-NN is logged as testcurve_knn_*.
        from analysis import knn_metrics
        dm, n = trainer.datamodule, trainer.datamodule.num_classes
        bank_h, bank_y = extract_features(pl_module.encoder, dm.bank_dataloader(self.bank_size))
        for split in splits:
            h, y = extract_features(pl_module.encoder, dm.split_dataloader(split, persistent=False))
            scores = knn_scores(bank_h, bank_y, h, n, self.k)  # [N, num_classes], the weighted votes
            name = "testcurve" if split == "test" and trainer.state.fn == "fit" else split
            metrics = {f"{name}_knn_top1": accuracy(scores, y, task="multiclass", num_classes=n, top_k=1),
                       f"{name}_knn_top5": accuracy(scores, y, task="multiclass", num_classes=n, top_k=5)}
            if trainer.state.fn != "fit":  # the kept checkpoint: the class metrics too
                metrics.update(knn_metrics(scores, y, n, prefix=f"{name}_knn_"))
            pl_module.log_dict(metrics, add_dataloader_idx=False)

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:  # the 2 check batches before training
            return
        # The test curve only while training, not in the trainer.validate() after it.
        with_test = trainer.datamodule.test_curve and trainer.state.fn == "fit"
        self._log_knn(trainer, pl_module, ("val", "test") if with_test else ("val",))

    def on_test_epoch_end(self, trainer, pl_module):
        self._log_knn(trainer, pl_module, ("test",))
